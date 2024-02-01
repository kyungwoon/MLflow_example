import os
import pandas as pd
from argparse import ArgumentParser
from sqlalchemy import create_engine
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 데이터베이스 연결 문자열 생성
db_string = "postgresql://postgres:postgres@localhost:5432/postgres"
db = create_engine(db_string)

# 데이터 가져오기
df = pd.read_sql("SELECT * FROM iris_data ORDER BY id DESC LIMIT 100", db)

# MLflow 환경 변수 설정
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniopassowrd"

# 데이터 전처리
X = df.drop(["id", "timestamp", "target"], axis="columns")
y = df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 모델 개발 및 훈련
model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
model_pipeline.fit(X_train, y_train)

# 예측 및 정확도 계산
train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)
train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)
print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# MLflow를 사용한 모델 저장
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="test_model")
args = parser.parse_args()
mlflow.set_experiment("test-exp")
signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
input_sample = X_train.iloc[:10]

with mlflow.start_run():
    mlflow.log_metrics({"train_acc": train_acc, "valid_acc": valid_acc})
    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path=args.model_name,
        signature=signature,
        input_example=input_sample,
    )

# 데이터 저장
df.to_csv("data.csv", index=False)


# python save_model.py --model-name "test_model"