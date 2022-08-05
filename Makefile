install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		
run:
	jupyter notebook 3DMM.ipynb
