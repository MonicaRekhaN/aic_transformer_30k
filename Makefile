install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	#python -m pytest -vv main.py

lint:
	pylint --disable=R,C main.py

deploy:
	echo "Deploying app"
	eb deploy Aictransformer8k-env

all: install lint test 
