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

!ifndef PAGE_BG_BMP
!define PAGE_BG_BMP "..\\..\\packaging\\windows\\assets\\generated\\page-bg.bmp"
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
Var BgImageHandle

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

  ${NSD_CreateBitmap} 0 0 100% 100% ""
  Pop $0
  ${NSD_SetStretchedBitmap} $0 "${PAGE_BG_BMP}" $BgImageHandle

  ${NSD_CreateLabel} 6u 8u 96% 52u ""
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x00070E1D

  ${NSD_CreateLabel} 14u 16u 58% 12u "VOLTA INSTALLER"
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x00070E1D

  ${NSD_CreateLabel} 14u 30u 58% 10u "Deterministic CLI runtime setup"
  Pop $0
  SetCtlColors $0 0x0093C6EA 0x00070E1D

  ${NSD_CreateLabel} 14u 42u 58% 10u "release: ${VERSION} // per-user install"
  Pop $0
  SetCtlColors $0 0x006EA7CD 0x00070E1D

  ${NSD_CreateLabel} 74% 16u 22% 36u ""
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x000A1530

  ${NSD_CreateLabel} 77% 21u 18% 10u "VOLTA"
  Pop $0
  SetCtlColors $0 0x00E8F4FF 0x000A1530

  ${NSD_CreateLabel} 77% 33u 18% 14u "DETERMINISTIC$\r$\nINSTALLER"
  Pop $0
  SetCtlColors $0 0x0088BBDD 0x000A1530

  ${NSD_CreateLabel} 8u 66u 96% 10u "Install location"
  Pop $0
  SetCtlColors $0 0x00DDEEFF 0x00101826

  ${NSD_CreateText} 8u 78u 78% 13u "$INSTDIR"
  Pop $InstallPathInput
  SetCtlColors $InstallPathInput 0x00EAF6FF 0x000E1B2E

  ${NSD_CreateButton} 88% 78u 11% 13u "Browse..."
  Pop $0
  ${NSD_OnClick} $0 OnBrowseInstallDir

  ${NSD_CreateCheckbox} 8u 98u 92% 12u "Add Volta binary directory to user PATH"
  Pop $CheckboxPath
  SetCtlColors $CheckboxPath 0x00EAF6FF 0x00101826
  ${NSD_Check} $CheckboxPath

  ${NSD_CreateCheckbox} 8u 114u 92% 12u "Run 'volta doctor --strict' after install"
  Pop $CheckboxDoctor
  SetCtlColors $CheckboxDoctor 0x00EAF6FF 0x00101826

  ${NSD_CreateLabel} 8u 132u 96% 30u ""
  Pop $0
  SetCtlColors $0 0x009BC5E5 0x00070E1D

  ${NSD_CreateLabel} 12u 138u 92% 20u ""
  Pop $SummaryLabel
  SetCtlColors $SummaryLabel 0x00B4D8F2 0x00070E1D

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
  ${NSD_SetText} $SummaryLabel "Install target selected above.$\r$\nNo admin rights are required.$\r$\nLogs: installer.log in install folder."
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
