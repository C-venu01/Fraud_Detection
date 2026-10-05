python model_selection/select_model.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python without_preprocessing/train.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python without_preprocessing/evaluate.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python with_preprocessing/train.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python with_preprocessing/evaluate.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python comparison/compare.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
