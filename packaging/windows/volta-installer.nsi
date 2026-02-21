!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"
!include "WinMessages.nsh"
!include "StrFunc.nsh"

!insertmacro StrStr
!insertmacro StrRep

!ifndef VERSION
!define VERSION "dev"
!endif

!ifndef VOLTA_BINARY
!define VOLTA_BINARY "..\\..\\target\\release\\volta.exe"
!endif

!ifndef OUT_DIR
!define OUT_DIR "..\\..\\dist\\release\\${VERSION}\\windows"
!endif

!ifndef HEADER_BMP
!define HEADER_BMP "..\\..\\packaging\\windows\\assets\\generated\\header.bmp"
!endif

!ifndef WELCOME_BMP
!define WELCOME_BMP "..\\..\\packaging\\windows\\assets\\generated\\welcome.bmp"
!endif

Unicode true
ManifestDPIAware true
Name "Volta ${VERSION}"
OutFile "${OUT_DIR}\\VoltaSetup-${VERSION}.exe"
InstallDir "$LOCALAPPDATA\\Volta\\bin"
RequestExecutionLevel user
BrandingText "VOLTA // deterministic installer"

VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "Volta CLI"
VIAddVersionKey "CompanyName" "Volta OSS"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileDescription" "Volta Installer"

Var CheckboxPath
Var CheckboxDoctor
Var OptionAddToPath
Var OptionRunDoctor
Var InstallLog

!define MUI_ABORTWARNING
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "${HEADER_BMP}"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${WELCOME_BMP}"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "${WELCOME_BMP}"

!insertmacro MUI_PAGE_WELCOME
Page custom OptionsPageCreate OptionsPageLeave
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "Install Volta" SEC_MAIN
  SetOutPath "$INSTDIR"
  File "/oname=volta.exe" "${VOLTA_BINARY}"

  StrCpy $InstallLog "$INSTDIR\\installer.log"
  Push "[volta-installer] install root: $INSTDIR"
  Call WriteLogLine

  ${If} $OptionAddToPath == ${BST_CHECKED}
    Call AddUserPath
    Push "[volta-installer] PATH updated for current user"
    Call WriteLogLine
  ${Else}
    Push "[volta-installer] PATH update skipped"
    Call WriteLogLine
  ${EndIf}

  Call VerifyVersion

  ${If} $OptionRunDoctor == ${BST_CHECKED}
    Call VerifyDoctor
  ${Else}
    Push "[volta-installer] doctor verification skipped"
    Call WriteLogLine
  ${EndIf}

  WriteUninstaller "$INSTDIR\\Uninstall Volta.exe"

  WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Volta" "DisplayName" "Volta CLI"
  WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Volta" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Volta" "Publisher" "Volta OSS"
  WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Volta" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Volta" "UninstallString" '"$INSTDIR\\Uninstall Volta.exe"'
  WriteRegDWORD HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Volta" "NoModify" 1
  WriteRegDWORD HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Volta" "NoRepair" 1
SectionEnd

Section "Uninstall"
  StrCpy $InstallLog "$INSTDIR\\installer.log"
  Push "[volta-installer] uninstall started"
  Call WriteLogLine

  Call RemoveUserPath

  Delete "$INSTDIR\\volta.exe"
  Delete "$INSTDIR\\Uninstall Volta.exe"
  Delete "$INSTDIR\\installer.log"
  RMDir "$INSTDIR"

  DeleteRegKey HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Volta"
SectionEnd

Function OptionsPageCreate
  nsDialogs::Create 1018
  Pop $0

  ${If} $0 == error
    Abort
  ${EndIf}

  ${NSD_CreateLabel} 0 0 100% 24u "VOLTA INSTALLER"
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x00060912

  ${NSD_CreateLabel} 0 18u 100% 24u "Deterministic CLI runtime setup"
  Pop $0
  SetCtlColors $0 0x0088BBDD 0x00060912

  ${NSD_CreateCheckbox} 0 50u 100% 12u "Add Volta binary directory to user PATH"
  Pop $CheckboxPath
  ${NSD_Check} $CheckboxPath

  ${NSD_CreateCheckbox} 0 68u 100% 12u "Run 'volta doctor --strict' after install"
  Pop $CheckboxDoctor

  ${NSD_CreateLabel} 0 92u 100% 56u "Install target: $LOCALAPPDATA\\Volta\\bin$\r$\nNo admin rights are required.$\r$\nLogs: $LOCALAPPDATA\\Volta\\bin\\installer.log"
  Pop $0
  SetCtlColors $0 0x007FA9C7 0x00060912

  nsDialogs::Show
FunctionEnd

Function OptionsPageLeave
  ${NSD_GetState} $CheckboxPath $OptionAddToPath
  ${NSD_GetState} $CheckboxDoctor $OptionRunDoctor
FunctionEnd

Function AddUserPath
  ReadRegStr $0 HKCU "Environment" "Path"
  StrCpy $1 "$0;"
  StrCpy $2 "$INSTDIR;"
  ${StrStr} $3 "$1" "$2"

  ${If} $3 == ""
    ${If} $0 == ""
      StrCpy $4 "$INSTDIR"
    ${Else}
      StrCpy $4 "$0;$INSTDIR"
    ${EndIf}

    WriteRegExpandStr HKCU "Environment" "Path" "$4"
    SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
  ${EndIf}
FunctionEnd

Function RemoveUserPath
  ReadRegStr $0 HKCU "Environment" "Path"

  ${If} $0 != ""
    ${StrRep} $0 $0 "$INSTDIR;" ""
    ${StrRep} $0 $0 ";$INSTDIR" ""
    ${StrRep} $0 $0 "$INSTDIR" ""
    WriteRegExpandStr HKCU "Environment" "Path" "$0"
    SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
  ${EndIf}
FunctionEnd

Function VerifyVersion
  nsExec::ExecToStack '"$INSTDIR\\volta.exe" version'
  Pop $0
  Pop $1

  ${If} $0 == "0"
    Push "[volta-installer] verify version: OK -> $1"
    Call WriteLogLine
  ${Else}
    Push "[volta-installer] verify version: FAIL -> $1"
    Call WriteLogLine
    MessageBox MB_ICONEXCLAMATION "Installed binary, but version check failed.$\r$\n$1"
  ${EndIf}
FunctionEnd

Function VerifyDoctor
  nsExec::ExecToStack '"$INSTDIR\\volta.exe" doctor --strict'
  Pop $0
  Pop $1

  ${If} $0 == "0"
    Push "[volta-installer] doctor strict: OK"
    Call WriteLogLine
  ${Else}
    Push "[volta-installer] doctor strict: WARN/FAIL -> $1"
    Call WriteLogLine
    MessageBox MB_ICONEXCLAMATION "volta doctor --strict reported issues.$\r$\n$1"
  ${EndIf}
FunctionEnd

Function WriteLogLine
  Exch $0
  FileOpen $1 "$InstallLog" a
  FileWrite $1 "$0$\r$\n"
  FileClose $1
FunctionEnd
