@echo off
echo ============================================================
echo MVP-Leadership Intelligence System - Directory Setup
echo ============================================================
echo.

REM Get the current directory (where this batch file is located)
set CURRENT_DIR=%~dp0

echo Setting up directory structure in: %CURRENT_DIR%
echo.

REM Create docs directory
if not exist "%CURRENT_DIR%docs" (
    mkdir "%CURRENT_DIR%docs"
    echo ✓ Created: docs\
) else (
    echo ✓ Already exists: docs\
)

REM Create examples directory
if not exist "%CURRENT_DIR%examples" (
    mkdir "%CURRENT_DIR%examples"
    echo ✓ Created: examples\
) else (
    echo ✓ Already exists: examples\
)

REM Create output directory
if not exist "%CURRENT_DIR%output" (
    mkdir "%CURRENT_DIR%output"
    echo ✓ Created: output\
) else (
    echo ✓ Already exists: output\
)

REM Create cache directory (for future performance optimization)
if not exist "%CURRENT_DIR%cache" (
    mkdir "%CURRENT_DIR%cache"
    echo ✓ Created: cache\
) else (
    echo ✓ Already exists: cache\
)

REM Create logs directory
if not exist "%CURRENT_DIR%logs" (
    mkdir "%CURRENT_DIR%logs"
    echo ✓ Created: logs\
) else (
    echo ✓ Already exists: logs\
)

REM Create config directory (for backup configurations)
if not exist "%CURRENT_DIR%config" (
    mkdir "%CURRENT_DIR%config"
    echo ✓ Created: config\
) else (
    echo ✓ Already exists: config\
)

echo.
echo ============================================================
echo Directory Setup Complete!
echo ============================================================
echo.
echo Your MVP-Leadership Intelligence System directory now has:
echo   📁 docs\          - Documentation files
echo   📁 examples\      - Sample output files
echo   📁 output\        - Generated intelligence reports
echo   📁 cache\         - Performance optimization cache
echo   📁 logs\          - System log files
echo   📁 config\        - Configuration backups
echo.
echo Next steps:
echo 1. Move SETUP-GUIDE.md to docs\ folder
echo 2. Move CUSTOMIZATION.md to docs\ folder
echo 3. Move sample output files to examples\ folder
echo 4. Update Config class in your script to use output\ folder
echo.
echo ============================================================
pause