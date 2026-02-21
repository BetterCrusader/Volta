!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"
!include "WinMessages.nsh"
!include "StrFunc.nsh"

${Using:StrFunc} StrStr
${Using:StrFunc} UnStrRep

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
VIAddVersionKey "LegalCopyright" "Copyright (c) Volta OSS"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileDescription" "Volta Installer"

Var CheckboxPath
Var CheckboxDoctor
Var InstallPathInput
Var SummaryLabel
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
  CreateDirectory "$INSTDIR"
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
  ReadRegStr $0 HKCU "Environment" "Path"
  ${If} $0 != ""
    ${UnStrRep} $0 $0 "$INSTDIR;" ""
    ${UnStrRep} $0 $0 ";$INSTDIR" ""
    ${UnStrRep} $0 $0 "$INSTDIR" ""
    WriteRegExpandStr HKCU "Environment" "Path" "$0"
    SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
  ${EndIf}

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

  ${NSD_CreateLabel} 0 0 100% 52u ""
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x00060912

  ${NSD_CreateLabel} 8u 7u 64% 12u "VOLTA INSTALLER"
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x00060912

  ${NSD_CreateLabel} 8u 22u 64% 10u "Deterministic CLI runtime setup"
  Pop $0
  SetCtlColors $0 0x0088BBDD 0x00060912

  ${NSD_CreateLabel} 8u 34u 64% 10u "release: ${VERSION} // per-user install"
  Pop $0
  SetCtlColors $0 0x00659EC0 0x00060912

  ${NSD_CreateLabel} 73% 6u 25% 38u ""
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x000A1530

  ${NSD_CreateLabel} 76% 11u 20% 10u "VOLTA"
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x000A1530

  ${NSD_CreateLabel} 76% 23u 20% 14u "DETERMINISTIC INSTALLER"
  Pop $0
  SetCtlColors $0 0x0088BBDD 0x000A1530

  ${NSD_CreateLabel} 0 60u 100% 10u "Install location"
  Pop $0
  SetCtlColors $0 0x004B637B 0x00F3F3F3

  ${NSD_CreateText} 0 72u 79% 12u "$INSTDIR"
  Pop $InstallPathInput

  ${NSD_CreateButton} 81% 72u 19% 12u "Browse..."
  Pop $0
  ${NSD_OnClick} $0 OnBrowseInstallDir

  ${NSD_CreateCheckbox} 0 93u 100% 12u "Add Volta binary directory to user PATH"
  Pop $CheckboxPath
  ${NSD_Check} $CheckboxPath

  ${NSD_CreateCheckbox} 0 111u 100% 12u "Run 'volta doctor --strict' after install"
  Pop $CheckboxDoctor

  ${NSD_CreateLabel} 0 132u 100% 42u ""
  Pop $0
  SetCtlColors $0 0x007FA9C7 0x00060912

  ${NSD_CreateLabel} 2% 136u 96% 34u ""
  Pop $SummaryLabel
  SetCtlColors $SummaryLabel 0x007FA9C7 0x00060912

  Call UpdateInstallSummary

  nsDialogs::Show
FunctionEnd

Function OptionsPageLeave
  ${NSD_GetText} $InstallPathInput $0
  ${If} $0 == ""
    MessageBox MB_ICONEXCLAMATION "Install location cannot be empty."
    Abort
  ${EndIf}

  StrCpy $INSTDIR "$0"
  ${NSD_GetState} $CheckboxPath $OptionAddToPath
  ${NSD_GetState} $CheckboxDoctor $OptionRunDoctor
FunctionEnd

Function OnBrowseInstallDir
  nsDialogs::SelectFolderDialog "Choose Volta install directory" "$INSTDIR"
  Pop $0

  ${If} $0 != "error"
    StrCpy $INSTDIR "$0"
    ${NSD_SetText} $InstallPathInput "$INSTDIR"
    Call UpdateInstallSummary
  ${EndIf}
FunctionEnd

Function UpdateInstallSummary
  ${NSD_SetText} $SummaryLabel "Install target: $INSTDIR$\r$\nNo admin rights are required.$\r$\nLogs: $INSTDIR\\installer.log"
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
