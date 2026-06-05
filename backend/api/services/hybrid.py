from typing import Optional

BERT_WEIGHT = 0.6
XGB_WEIGHT  = 0.4
XGB_GATE    = 0.65   # XGBoost bu değerin altındaysa blending'e katılmaz (precision=0.25 sorunu)
THRESHOLD   = 0.45


def compute(
    bert_score: Optional[float],
    xgb_score:  Optional[float],
) -> tuple[float, float, float]:
    b = bert_score if bert_score is not None else 0.0
    x = xgb_score  if xgb_score  is not None else 0.0

    if bert_score is None and xgb_score is None:
        return 0.0, 0.0, 0.0
    if bert_score is None:
        final = x
    elif xgb_score is None:
        final = b
    else:
        xgb_contrib     = x * XGB_WEIGHT if x >= XGB_GATE else 0.0
        bert_weight_adj = 1.0 if xgb_contrib == 0.0 else BERT_WEIGHT
        final = b * bert_weight_adj + xgb_contrib

    return round(final, 4), round(b, 4), round(x, 4)


def label(final_score: float) -> str:
    if final_score > THRESHOLD:
        return "🚨 RİSKLİ / SİBER ZORBALIK"
    return "✅ GÜVENLİ / TEMİZ"
