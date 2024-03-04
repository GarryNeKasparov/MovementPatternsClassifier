import base64
import io

import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from mpproject.src.app.generate_test import gen_data
from mpproject.src.data.utils import (
    compute_velocity,
    df_replace_outliers_by_object,
)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
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
async def generate_dataframe():
    """
    Обработка POST запроса для генерации DataFrame.
    """
    df = gen_data(20)
    df = df_replace_outliers_by_object((df, "15T")).reset_index()
    df = compute_velocity(df, "velocity_c")

    idx = ((df["velocity"].isna()) | (df["velocity"] > 40)) & (df["velocity_c"] <= 40)
    df.loc[idx, "velocity"] = df.loc[idx, "velocity_c"]
    df["velocity"] = df["velocity"].interpolate()
    plt.plot(df["x"], df["y"], marker="o", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot of x vs y")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    df_html = df.to_html()
    html_content = f"""
    <html>
        <head>
            <title>Generated DataFrame</title>
        </head>
        <body>
            <h1>Generated DataFrame</h1>
            <form action="/generated" method="post">
                <button type="submit">Regenerate</button>
            </form>
            <table border="1" style="display:inline-block">
                <tr>
                    <td>
                        {df_html}
                    </td>
                    <td>
                        <img src="data:image/png;base64, {plot_base64}">
                    </td>
                </tr>
            </table>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
