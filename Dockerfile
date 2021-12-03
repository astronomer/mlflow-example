FROM quay.io/astronomer/ap-airflow:2.1.0-2-buster-onbuild

ENV AIRFLOW__CORE__XCOM_BACKEND=include.gcs_xcom_backend.GCSXComBackend

USER root

# Required for some ML/DS dependencies
RUN apt-get update -y
RUN apt-get install libgomp1 -y
RUN apt-get install -y git

USER astro