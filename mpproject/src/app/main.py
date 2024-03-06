import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import uvicorn
from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from mpproject.models.data_utils import get_model
from mpproject.models.predict import get_predictions
from mpproject.src.app.generate_test import gen_data
from mpproject.src.data.utils import (
    compute_velocity,
    df_replace_outliers_by_object,
)

templates = Jinja2Templates(directory="mpproject/src/app/static")


class Data(BaseModel):
    df: str
    modelname: str


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Отображает страницу с кнопкой и, при нажатии, сгенерирует DataFrame и отобразит его.
    """
    html_content = """
    <html>
        <head>
            <title>Generate DataFrame</title>
        </head>
        <body>
            <h1>Generate DataFrame</h1>
            <form action="/generated" method="post">
                <button type="submit">Generate DataFrame</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/generated")
async def generate_dataframe(request: Request):
    """
    Обработка POST запроса для генерации DataFrame.
    """
    df = gen_data(20)
    df = df_replace_outliers_by_object((df, "15min")).reset_index()
    df = compute_velocity(df, "velocity_c")

    idx = ((df["velocity"].isna()) | (df["velocity"] > 40)) & (df["velocity_c"] <= 40)
    df.loc[idx, "velocity"] = df.loc[idx, "velocity_c"]
    df["velocity"] = df["velocity"].interpolate()
    return templates.TemplateResponse(
        request=request, name="generated.html", context={"df": df}
    )


@app.post("/predict")
async def predict(data: Data):
    """
    Возвращает предсказания для сгенерированных данных.
    """
    df = pd.read_html(data.df)[0]
    try:
        model = get_model(data.modelname)
        model.eval()
        preds = get_predictions(model, df)
        df["prediction"] = preds
        plt.plot(df["x"], df["y"], linestyle="dashed")
        sns.scatterplot(data=df, x="x", y="y", hue=preds, zorder=2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Маршрут и предсказание типа")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return {
            "df": df.to_html(
                columns=[
                    "datetime",
                    "x",
                    "y",
                    "velocity",
                    "velocity_c",
                    "target",
                    "prediction",
                ]
            ),
            "plot": plot_base64,
            "status": "ok",
        }
    except AssertionError as msg:
        return {"message": str(msg), "status": "err"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
