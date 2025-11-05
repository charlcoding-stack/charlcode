; Charl Language Installer Script for Inno Setup
;
; This creates a professional Windows installer (.exe) that:
; - Downloads or includes the pre-compiled binary
; - Adds Charl to system PATH
; - Installs VS Code extension (optional)
; - Creates Start Menu shortcuts
;
; Download Inno Setup from: https://jrsoftware.org/isdl.php
; Compile this script with Inno Setup Compiler to create charl-setup.exe

#define MyAppName "Charl"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Charl Team"
#define MyAppURL "https://charlbase.org"
#define MyAppExeName "charl.exe"

[Setup]
; Basic Information
AppId={{8B7F3D4E-9A2C-4F1E-B8D6-5C3E2A1F9B7E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/docs
AppUpdatesURL={#MyAppURL}/downloads
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir=..\releases
OutputBaseFilename=charl-setup-{#MyAppVersion}-windows-x64
SetupIconFile=..\assets\charl-icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

; Modern UI
DisableProgramGroupPage=yes
DisableWelcomePage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "Add to PATH (recommended)"; GroupDescription: "System Integration:"; Flags: checkedonce
Name: "vscodeext"; Description: "Install VS Code extension (if VS Code is detected)"; GroupDescription: "IDE Integration:"; Flags: checkedonce

[Files]
; Main executable (you need to compile charl.exe first and place it here)
Source: "..\target\release\charl.exe"; DestDir: "{app}"; Flags: ignoreversion
; VS Code extension
Source: "..\vscode-charl\*"; DestDir: "{app}\vscode-extension"; Flags: ignoreversion recursesubdirs createallsubdirs
; Documentation
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\WINDOWS_BUILD_ISSUES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:ProgramOnTheWeb,{#MyAppName}}"; Filename: "{#MyAppURL}"
Name: "{group}\Documentation"; Filename: "{app}\README.md"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Optional: Run charl --version to verify installation
Filename: "{app}\{#MyAppExeName}"; Parameters: "--version"; Flags: nowait postinstall skipifsilent runhidden
; Show final page
Filename: "{app}\README.md"; Description: "View README"; Flags: postinstall shellexec skipifsilent unchecked

[Code]
var
  VSCodePath: String;

// Add to PATH function
procedure AddToPath();
var
  OldPath, NewPath: String;
begin
  if RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OldPath) then
  begin
    NewPath := ExpandConstant('{app}');
    if Pos(Uppercase(NewPath), Uppercase(OldPath)) = 0 then
    begin
      NewPath := OldPath + ';' + NewPath;
      RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', NewPath);
    end;
  end
  else
  begin
    RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', ExpandConstant('{app}'));
  end;
end;

// Remove from PATH on uninstall
procedure RemoveFromPath();
var
  OldPath, NewPath, AppPath: String;
  P: Integer;
begin
  AppPath := ExpandConstant('{app}');
  if RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OldPath) then
  begin
    NewPath := OldPath;
    P := Pos(';' + Uppercase(AppPath), Uppercase(NewPath));
    if P = 0 then
      P := Pos(Uppercase(AppPath) + ';', Uppercase(NewPath));
    if P = 0 then
      P := Pos(Uppercase(AppPath), Uppercase(NewPath));
    if P > 0 then
    begin
      Delete(NewPath, P, Length(AppPath) + 1);
      RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', NewPath);
    end;
  end;
end;

// Check if VS Code is installed
function FindVSCode(): Boolean;
begin
  Result := False;
  VSCodePath := '';

  // Check common locations
  if FileExists(ExpandConstant('{pf}\Microsoft VS Code\Code.exe')) then
  begin
    VSCodePath := ExpandConstant('{pf}\Microsoft VS Code\Code.exe');
    Result := True;
  end
  else if FileExists(ExpandConstant('{localappdata}\Programs\Microsoft VS Code\Code.exe')) then
  begin
    VSCodePath := ExpandConstant('{localappdata}\Programs\Microsoft VS Code\Code.exe');
    Result := True;
  end;
end;

// Install VS Code extension
procedure InstallVSCodeExtension();
var
  ExtPath: String;
  ResultCode: Integer;
begin
  if FindVSCode() then
  begin
    ExtPath := ExpandConstant('{app}\vscode-extension');
    // Copy extension to VS Code extensions folder
    if DirExists(ExpandConstant('{userappdata}\Code\User\extensions')) then
    begin
      if not DirExists(ExpandConstant('{userappdata}\Code\User\extensions\charl-lang.charl-1.0.0')) then
      begin
        // Use robocopy or manual copy
        Exec('cmd.exe', '/c xcopy "' + ExtPath + '" "' +
             ExpandConstant('{userappdata}\Code\User\extensions\charl-lang.charl-1.0.0') +
             '" /E /I /H /Y', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      end;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Add to PATH if task selected
    if IsTaskSelected('addtopath') then
      AddToPath();

    // Install VS Code extension if task selected
    if IsTaskSelected('vscodeext') then
      InstallVSCodeExtension();
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    RemoveFromPath();
  end;
end;

// Custom wizard page for additional info
procedure InitializeWizard();
var
  InfoPage: TOutputMsgMemoWizardPage;
begin
  InfoPage := CreateOutputMsgMemoPage(wpWelcome,
    'Information', 'Please read the following important information before continuing.',
    'Charl is an AI/ML programming language designed for high-performance machine learning.'#13#13 +
    'This installer will:'#13 +
    '  • Install Charl compiler and CLI tools'#13 +
    '  • Optionally add Charl to your PATH'#13 +
    '  • Optionally install VS Code extension'#13#13 +
    'For more information, visit: https://charlbase.org');
end;
