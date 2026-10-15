run:
	python run.py

run-all:
	python run.py --pipeline

setup:
	bash infra/setup_hadoop.sh --local-fallback
	bash infra/setup_kafka.sh
	python infra/check_infra.py --local-fallback

ingest:
	python spark/ingest.py --local-fallback

features:
	python spark/feature_engineering.py --local-fallback

train:
	python spark/train_pipeline.py --local-fallback

serve:
	uvicorn api.server:app --reload --port 8000

stream:
	python stream/kafka_producer.py

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

