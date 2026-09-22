import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# 기본 페이지 설정 (브라우저 탭 제목 + 아이콘)
# ---------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# 데이터 불러오기 (캐시 사용)
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------
# 앱 제목
# ---------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.write("이 앱은 뇌졸중 관련 건강 데이터를 살펴보고 예측 모델을 만들어보는 실습용 앱입니다.")

st.markdown("---")

# ---------------------------------------------------
# 소개 화면
# ---------------------------------------------------
st.header("📌 데이터 소개")

total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = round(stroke_count / total_people * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중(stroke=1) 인원", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio} %")

st.markdown("---")

# ---------------------------------------------------
# 열 설명 표 (우리말 뜻은 직접 채워 넣을 수 있도록 빈 칸)
# ---------------------------------------------------
st.header("📋 열(컬럼) 설명표")
st.caption("※ '우리말 뜻' 칸은 비어 있습니다. 교재를 참고하여 직접 채워 넣어 보세요.")

column_info = []
for col in df.columns:
    dtype = df[col].dtype
    n_unique = df[col].nunique(dropna=True)
    n_missing = df[col].isnull().sum()

    if dtype == "object" or n_unique <= 10:
        unique_vals = df[col].dropna().unique()
        value_kind = ", ".join(map(str, sorted(unique_vals, key=lambda x: str(x))))
    else:
        value_kind = f"숫자 (예: {df[col].min()} ~ {df[col].max()})"

    column_info.append({
        "열 이름": col,
        "우리말 뜻": "",
        "값의 종류": value_kind,
        "빈 값 개수": n_missing
    })

info_df = pd.DataFrame(column_info)

edited_info_df = st.data_editor(
    info_df,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    key="column_info_editor"
)

st.markdown("---")

# ---------------------------------------------------
# 데이터 미리보기 (처음 5줄)
# ---------------------------------------------------
st.header("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# 데이터 출처 (직접 작성하는 자리)
# ---------------------------------------------------
st.header("📚 데이터 출처")

source_text = st.text_area(
    "아래에 교재에 나온 데이터 출처를 적어보세요.",
    value="",
    height=100,
    placeholder="예: 이 데이터는 ... 에서 가져왔습니다."
)

if source_text:
    st.success("출처가 입력되었습니다 ✅")
    st.write(source_text)
else:
    st.info("아직 출처가 입력되지 않았습니다.")
