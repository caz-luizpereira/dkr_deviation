# More info: https://flask.palletsprojects.com/en/2.0.x/quickstart/

# In CLI flask first time running commands in current project folder 

  # activate enviroment variables
    
    virtualenv generateweighins
  
  # activate the enviroment proposed

    generateweighins\Scripts\activate #PowerShell

  # install the Flask micro-framework (only if it is your first time)

    pip install Flask
    pip install Flask-MQTT
    pip install Flask-Cors
    pip install -U scikit-learn

  # set development state in the venv

   # Windows:
      set FLASK_ENV=development
      set AUTHLIB_INSECURE_TRANSPORT=true

    or 

  # Linux:
      export FLASK_ENV=development 
      export AUTHLIB_INSECURE_TRANSPORT=true

  # Run with host machine IP

    flask run --host 0.0.0.0

