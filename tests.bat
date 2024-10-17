:: Tests require DATA_DIR and PRETRAINED_DIR env variables

coverage run -m pytest tests
coverage report -m
coverage html 
