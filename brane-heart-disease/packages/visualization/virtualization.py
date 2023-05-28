import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Any
from plotly.subplots import make_subplots


def feature_importance_rf(model: Any, feature_names: List):
    imp = model.feature_importances_
    imp_sorted = pd.Series(imp, index=feature_names).sort_values(ascending=True)
    # sns.barplot(y=imp_sorted.index, x=imp_sorted)
    # plt.show()
    df = pd.DataFrame(
        {"Feature Name": imp_sorted.index, "Feature Importance": imp_sorted}
    )
    fig = px.bar(
        df,
        x="Feature Importance",
        y="Feature Name",
        color="Feature Importance",
        orientation="h",
    )
    fig.update_layout(title=f"<b>Feature Importance</b>")
    fig.update_coloraxes(showscale=False)
    fig.write_image("feature_importance_rf.png")


def positive_ratio_histogram(df, feature_name, label_name, postive_value=1):
    postive_cnt = (
        df[df[label_name] == postive_value].groupby([feature_name])[label_name].count()
    )
    total_cnt = df.groupby([feature_name])[label_name].count()
    postive_ratio = postive_cnt / total_cnt
    feature_category = total_cnt.index
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=feature_category,
            y=postive_ratio,
            marker=dict(color=["lightgray"] * 9 + ["darkred"] * 4),
            name=f"The ratio of {feature_name} in {label_name}",
        ),
        secondary_y=False,
    )
    fig.update_layout(title=f"<b>The ratio of {feature_name} in {label_name}</b>")
    fig.update_yaxes(title="Ratio", rangemode="tozero", secondary_y=False)
    fig.update_xaxes(title=feature_name)
    fig.write_image(f"postive_ratio_{feature_name}.png")
