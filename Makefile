# Build two explicit exes with MSVC (nmake)
all: Dither.exe DitherAdv.exe

Dither.exe: Dither.cpp
    cl /nologo /EHsc /std:c++17 /c Dither.cpp
    link /nologo /OUT:Dither.exe Dither.obj

DitherAdv.exe: DitherAdv.cpp
    cl /nologo /EHsc /std:c++17 /c DitherAdv.cpp
    link /nologo /OUT:DitherAdv.exe DitherAdv.obj


clean:
    del /Q *.obj *.exe
