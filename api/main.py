"""FastAPI backend: tahmin servisi ve statik arayüz sunumu."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.predict import SalesPredictor, get_predictor

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

_state: dict[str, Any] = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Model süreç başlarken bir kez yüklenir."""
    try:
        _state["predictor"] = get_predictor()
        m = _state["predictor"].metrics.get("classification", {})
        print(f"[api] Model yüklendi. Test macro-F1 = {m.get('macro_f1', '?')}")
    except Exception as exc:
        _state["error"] = str(exc)
        print(f"[api] UYARI: model yüklenemedi -> {exc}")
    yield


app = FastAPI(
    title="Oyun Satış Başarısı Tahminleme API",
    description=(
        "Bir oyunun adı, platformu, türü, yayıncısı, çıkış yılı ve Metacritic "
        "puanlarından dünya geneli satışını ve satış segmentini birlikte "
        "tahmin eder. Regresyon ve sınıflandırma çıktıları tutarlılık motoru "
        "ile birleştirilir, bu yüzden asla çelişmezler."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _predictor() -> SalesPredictor:
    if _state["predictor"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model hazır değil. Önce `python -m src.train` çalıştırın. "
                   f"({_state['error']})")
    return _state["predictor"]


def _check_year(p: SalesPredictor, year: int) -> None:
    """Çıkış yılı veride gözlenen aralığın dışındaysa isteği reddeder."""
    if not (p.year_min <= year <= p.year_max):
        raise HTTPException(
            status_code=422,
            detail=(f"Çıkış yılı {year} desteklenmiyor. Veri seti yalnızca "
                    f"{p.year_min} ile {p.year_max} arası oyunları içeriyor."))


class GameRequest(BaseModel):
    name: str = Field(
        "", max_length=200, examples=["Call of Duty: Modern Warfare 4"],
        description=("Oyunun adı. Adından seri çıkarılır ve serinin ÖNCEKİ "
                     "oyunlarının performansı tahmine dahil edilir. Boş "
                     "bırakılırsa yeni IP sayılır."))
    platform: str = Field(..., examples=["PS4"], description="Çıkış platformu")
    genre: str = Field(..., examples=["Shooter"], description="Oyun türü")
    publisher: str = Field(..., examples=["Activision"], description="Yayıncı")
    year: int = Field(..., examples=[2014],
                      description=("Çıkış yılı. Veri setinde gözlenen aralığın "
                                   "dışında bir değer 422 ile reddedilir."))
    rating: str = Field("Unknown", examples=["M"],
                        description="ESRB yaş sınıflandırması (E, E10+, T, M, ...)")

    critic_score: float | None = Field(
        None, ge=0, le=100, examples=[83],
        description="Metacritic kritik puanı (0-100). En güçlü tek sinyal.")
    user_score: float | None = Field(
        None, ge=0, le=10, examples=[5.7],
        description="Metacritic kullanıcı puanı (0-10)")

    @field_validator("platform", "genre", "publisher")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Boş bırakılamaz")
        return v

    def to_record(self) -> dict[str, Any]:
        return {
            "Name": self.name, "Platform": self.platform, "Genre": self.genre,
            "Publisher": self.publisher, "Year_of_Release": self.year,
            "Rating": self.rating, "Critic_Score": self.critic_score,
            "User_Score": self.user_score,
        }


class BatchRequest(BaseModel):
    games: list[GameRequest] = Field(..., min_length=1, max_length=200)


@app.get("/api/health", tags=["sistem"])
def health() -> dict[str, Any]:
    ready = _state["predictor"] is not None
    return {"status": "ok" if ready else "model_yok",
            "model_ready": ready, "error": _state["error"]}


@app.get("/api/options", tags=["arayüz"])
def options() -> dict[str, Any]:
    """Arayüzdeki tüm form alanlarını dolduran seçenek listeleri."""
    return _predictor().options


@app.get("/api/metrics", tags=["sistem"])
def metrics() -> dict[str, Any]:
    """Test performansı, iki protokol, çapraz doğrulama ve tutarlılık raporu."""
    return _predictor().metrics


@app.get("/api/segments", tags=["arayüz"])
def segments() -> list[dict[str, Any]]:
    """Segment tanımları ve satış sınırları."""
    return _predictor().options["segments"]


@app.get("/api/franchise", tags=["arayüz"])
def franchise(name: str, year: int) -> dict[str, Any]:
    """Bir oyun adından çıkarılan seriyi ve o yıldan ÖNCEKİ geçmişini döndürür."""
    import math

    from src.data import franchise_key

    p = _predictor()
    key = franchise_key(name)
    count, mean_log, max_log, gap = p.fb.franchise_history.lookup(key, year)

    def back(v):
        return None if v is None or math.isnan(v) else round(math.expm1(v), 3)

    return {
        "input": name,
        "franchise_key": key,
        "prior_games": int(count),
        "prior_mean_millions": back(mean_log),
        "prior_best_millions": back(max_log),
        "years_since_last": None if gap is None or math.isnan(gap) else int(gap),
    }


@app.post("/api/predict", tags=["tahmin"])
def predict(req: GameRequest) -> dict[str, Any]:
    """Tek bir oyun için satış tahmini."""
    p = _predictor()
    _check_year(p, req.year)
    try:
        result = p.predict([req.to_record()])[0]
    except AssertionError as exc:
        raise HTTPException(500, f"Tutarlılık hatası: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return result.to_dict()


@app.post("/api/predict/batch", tags=["tahmin"])
def predict_batch(req: BatchRequest) -> dict[str, Any]:
    """Toplu tahmin, senaryo karşılaştırması için."""
    p = _predictor()
    for g in req.games:
        _check_year(p, g.year)
    try:
        results = p.predict([g.to_record() for g in req.games], explain=False)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"count": len(results), "results": [r.to_dict() for r in results]}


@app.get("/api/critic-curve", tags=["tahmin"])
def critic_curve(name: str = "", platform: str = "PS4", genre: str = "Shooter",
                 publisher: str = "Activision", year: int = 2014,
                 rating: str = "M", user_score: float | None = None
                 ) -> dict[str, Any]:
    """Kritik puanına göre tahmini satış eğrisi."""
    p = _predictor()
    _check_year(p, year)

    scores = list(range(30, 101, 5))
    records = [{
        "Name": name, "Platform": platform, "Genre": genre,
        "Publisher": publisher, "Year_of_Release": year, "Rating": rating,
        "Critic_Score": s, "User_Score": user_score,
    } for s in scores]

    try:
        results = p.predict(records, explain=False)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    return {
        "platform": platform,
        "points": [
            {"critic_score": s, "sales": r.predicted_sales,
             "segment": r.segment, "segment_index": r.segment_index}
            for s, r in zip(scores, results)
        ],
    }


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
