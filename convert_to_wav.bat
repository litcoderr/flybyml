cd C:\Users\litco\Desktop\atc_voice_pack

@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Directory where the converted files will be saved
SET "output_dir=..\atc_voice_pack_wav"

REM Create the output directory if it does not exist
IF NOT EXIST "%output_dir%" mkdir "%output_dir%"

REM Loop through all .m4a files in the current directory
FOR %%A IN (*.m4a) DO (
    REM Extract the filename without the extension
    SET "filename=%%~nA"

    REM Convert the file
    ffmpeg -i "%%A" "%output_dir%\!filename!.wav"
)

echo Conversion completed.
ENDLOCAL