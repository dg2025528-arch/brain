import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# 기본 페이지 설정
# ---------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🧪",
    layout="wide"
)

# ---------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("🧪 분류 모델 만들기")
st.write("나이, 혈당, 체질량지수, 고혈압, 심장병 정보를 이용해 뇌졸중을 예측하는 모델을 만들어봅니다.")

st.markdown("---")

# ---------------------------------------------------
# 열 이름 ↔ 우리말 이름 매핑
# ---------------------------------------------------
col_kor = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
kor_col = {v: k for k, v in col_kor.items()}

all_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi 제외

# ---------------------------------------------------
# 1. 속성 선택
# ---------------------------------------------------
st.header("1️⃣ 입력으로 사용할 속성 고르기")

selected_kor = st.multiselect(
    "모델의 입력(설명 변수)으로 사용할 속성을 고르세요.",
    options=[col_kor[c] for c in all_features],
    default=[col_kor[c] for c in default_features]
)

selected_features = [kor_col[k] for k in selected_kor]

if len(selected_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 골라주세요. 그림을 그리려면 최소 2개의 속성이 필요합니다.")
    st.stop()

st.markdown("---")

# ---------------------------------------------------
# 2. 데이터 준비 (10명씩 묶어 앞 3명 테스트, 뒤 7명 학습)
# ---------------------------------------------------
st.header("2️⃣ 데이터 준비")

work_df = df.copy()
work_df = work_df.sort_values("id").reset_index(drop=True)

# 10명씩 묶어서 그룹 번호 부여
work_df["group_index"] = work_df.index % 10

test_mask = work_df["group_index"] < 3
train_mask = ~test_mask

train_df = work_df[train_mask].copy()
test_df = work_df[test_mask].copy()

st.write(f"- 학습용 사람 수: **{len(train_df):,} 명**")
st.write(f"- 테스트용 사람 수: **{len(test_df):,} 명**")

# bmi 결측치 처리 (선택된 경우에만, 학습용 중앙값으로)
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.write(f"- 체질량지수(bmi) 결측치는 학습용 중앙값 **{bmi_median:.2f}** 로 채웠습니다.")

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

st.markdown("---")

# ---------------------------------------------------
# 3. 모델 학습
# ---------------------------------------------------
st.header("3️⃣ 모델 학습")

# 로지스틱 회귀
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# 의사결정트리 (깊이 3, 리프 최소 5명, 난수 고정)
tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

# 더미 모델 (다수결)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# 정확도 계산
def get_accuracies(model):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model)
tree_train_acc, tree_test_acc = get_accuracies(tree_model)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model)

# ---------------------------------------------------
# 정확도 카드
# ---------------------------------------------------
st.subheader("모델별 정확도")

card1, card2, card3 = st.columns(3)

with card1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{log_test_acc:.3f}"
    )
    st.caption(f"훈련 정확도: {log_train_acc:.3f}  ·  테스트 정확도: {log_test_acc:.3f}")

with card2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{tree_test_acc:.3f}"
    )
    st.caption(f"훈련 정확도: {tree_train_acc:.3f}  ·  테스트 정확도: {tree_test_acc:.3f}")

with card3:
    st.metric(
        label="다수결 모델(입력을 보지 않는 모델)",
        value=f"{dummy_test_acc:.3f}"
    )
    st.caption(f"훈련 정확도: {dummy_train_acc:.3f}  ·  테스트 정확도: {dummy_test_acc:.3f}")

st.markdown("---")

# ---------------------------------------------------
# 4. 산점도 + 결정 경계
# ---------------------------------------------------
st.header("4️⃣ 산점도와 결정 경계")

axis_kor = st.multiselect(
    "그림의 가로축과 세로축으로 사용할 속성 2개를 고르세요.",
    options=selected_kor,
    default=selected_kor[:2],
    max_selections=2
)

if len(axis_kor) != 2:
    st.warning("⚠️ 가로축과 세로축으로 사용할 속성을 정확히 2개 골라주세요.")
    st.stop()

x_col = kor_col[axis_kor[0]]
y_col = kor_col[axis_kor[1]]

other_features = [f for f in selected_features if f not in [x_col, y_col]]

# 다른 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {}
for f in other_features:
    fixed_values[f] = test_df[f].median()

if fixed_values:
    fixed_text = ", ".join([f"{col_kor[f]} = {v:.2f}" for f, v in fixed_values.items()])
    st.write(f"📌 그림에 나타나지 않는 속성은 다음 값으로 고정했습니다: **{fixed_text}**")
else:
    st.write("📌 고른 속성이 2개뿐이라 고정할 속성이 없습니다.")

# 산점도용 데이터
scatter_df = test_df.copy()
scatter_df["실제 뇌졸중 여부"] = scatter_df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

fig_scatter = go.Figure()

for label, color in [("뇌졸중 없음", "blue"), ("뇌졸중 있음", "red")]:
    sub = scatter_df[scatter_df["실제 뇌졸중 여부"] == label]
    fig_scatter.add_trace(go.Scatter(
        x=sub[x_col], y=sub[y_col],
        mode="markers",
        name=label,
        marker=dict(color=color, size=7, opacity=0.6)
    ))

# ---------------------------------------------------
# 결정 경계 계산을 위한 격자 생성
# ---------------------------------------------------
x_min, x_max = test_df[x_col].min(), test_df[x_col].max()
y_min, y_max = test_df[y_col].min(), test_df[y_col].max()

x_range = np.linspace(x_min, x_max, 200)
y_range = np.linspace(y_min, y_max, 200)
xx, yy = np.meshgrid(x_range, y_range)

grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for f, v in fixed_values.items():
    grid_df[f] = v

grid_df = grid_df[selected_features]  # 학습 시 순서와 맞추기

# 트리 모델의 배경색 (결정 영역)
tree_pred_grid = tree_model.predict(grid_df).reshape(xx.shape)

fig_scatter.add_trace(go.Contour(
    x=x_range, y=y_range, z=tree_pred_grid,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "blue"], [1, "red"]],
    contours=dict(showlines=False),
    name="의사결정트리 영역",
    hoverinfo="skip"
))

# ---------------------------------------------------
# 로지스틱 회귀 0.5 확률 경계선
# ---------------------------------------------------
log_prob_grid = log_model.predict_proba(grid_df)[:, 1].reshape(xx.shape)

# 0.5 등고선만 추출하기 위해 Contour 사용
fig_scatter.add_trace(go.Contour(
    x=x_range, y=y_range, z=log_prob_grid,
    showscale=False,
    contours=dict(
        start=0.5, end=0.5, size=0.1,
        coloring="lines"
    ),
    line=dict(color="black", width=3),
    name="로지스틱 회귀 경계선(0.5)",
    hoverinfo="skip"
))

# 경계선이 그림 안에 있는지 확인
in_range = ((log_prob_grid >= 0.49) & (log_prob_grid <= 0.51)).any()
if not in_range:
    st.info("ℹ️ 로지스틱 회귀의 0.5 확률 경계선이 이 그림의 범위 밖에 있어 보이지 않습니다.")

fig_scatter.update_layout(
    title="테스트 데이터 산점도와 결정 경계",
    xaxis_title=col_kor[x_col],
    yaxis_title=col_kor[y_col],
    legend_title="구분"
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# 5. 의사결정트리 가지 그림
# ---------------------------------------------------
st.header("5️⃣ 의사결정트리 가지 그림")

def build_dot(tree, feature_names, kor_names):
    tree_ = tree.tree_
    feature_name = [
        kor_names[feature_names[i]] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]

    dot_lines = ["digraph Tree {", 'node [shape=box, fontname="Malgun Gothic"];']

    leaf_count = [0]
    leaf_no_count = [0]
    used_features = set()

    def recurse(node, depth):
        n_samples = int(tree_.n_node_samples[node])
        value = tree_.value[node][0]
        n_stroke = int(value[1]) if len(value) > 1 else 0
        ratio = n_stroke / n_samples if n_samples > 0 else 0

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # 내부 노드 (질문 있음)
            name = feature_name[node]
            threshold = tree_.threshold[node]
            used_features.add(name)

            label = f"{name} <= {threshold:.2f}\\n사람 수: {n_samples}\\n뇌졸중: {n_stroke}명\\n비율: {ratio:.2f}"
            dot_lines.append(f'{node} [label="{label}", style=filled, fillcolor="white"];')

            left = tree_.children_left[node]
            right = tree_.children_right[node]

            recurse(left, depth + 1)
            recurse(right, depth + 1)

            dot_lines.append(f'{node} -> {left} [label="예"];')
            dot_lines.append(f'{node} -> {right} [label="아니요"];')
        else:
            # 리프 노드 (답을 내는 마디)
            pred = int(np.argmax(value))
            pred_label = "뇌졸중 있음" if pred == 1 else "뇌졸중 없음"
            color = "#ffcccc" if pred == 1 else "#cce5ff"

            leaf_count[0] += 1
            if pred == 0:
                leaf_no_count[0] += 1

            label = f"[답: {pred_label}]\\n사람 수: {n_samples}\\n뇌졸중: {n_stroke}명\\n비율: {ratio:.2f}"
            dot_lines.append(f'{node} [label="{label}", style=filled, fillcolor="{color}"];')

    recurse(0, 0)
    dot_lines.append("}")
    return "\n".join(dot_lines), leaf_count[0], leaf_no_count[0], used_features

dot_str, total_leaf, no_leaf, used_feats = build_dot(tree_model, selected_features, col_kor)

st.graphviz_chart(dot_str)

st.markdown("---")

# ---------------------------------------------------
# 6. 트리 요약 설명
# ---------------------------------------------------
st.header("6️⃣ 트리 요약")

st.write(f"- 답을 내는 마디(리프 노드)는 모두 **{total_leaf} 칸**이고, 그중 **{no_leaf} 칸**이 '뇌졸중 없음'이라고 답합니다.")

if used_feats:
    used_feats_str = ", ".join(sorted(used_feats))
    st.write(f"- 고른 속성 가운데 이 나무가 실제로 물은 속성은 **{used_feats_str}** 입니다.")
else:
    st.write("- 이 나무는 어떤 속성도 실제로 묻지 않았습니다. (모든 사람에게 같은 답을 합니다.)")
