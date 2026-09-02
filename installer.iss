#define AppVersion GetEnv("APP_VERSION")

[Setup]
AppName=SmartImage Archiver
AppVersion={#AppVersion}
VersionInfoVersion={#AppVersion}
VersionInfoProductVersion={#AppVersion}
DefaultDirName={localappdata}\SmartImageArchiver
OutputDir=Output
OutputBaseFilename=SmartImageArchiverSetup
Compression=lzma2
SolidCompression=yes

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Files]
Source: "SmartImageArchiver.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "unrar\UnRAR.exe"; DestDir: "{app}"

[Registry]
; メインメニュー
Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver"; \
    ValueName: "MUIVerb"; ValueType: string; \
    ValueData: "SmartImage Archiver"; \
    Flags: uninsdeletekey

; ZIP / RAR / 7z に対応
Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver"; \
    ValueName: "AppliesTo"; ValueType: string; \
    ValueData: "System.FileExtension:=.zip OR System.FileExtension:=.rar OR System.FileExtension:=.7z OR System.FileExtension:=.cbz OR System.FileExtension:=.cbr OR System.FileExtension:=.cb7"

Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver"; \
    ValueName: "ExtendedSubCommandsKey"; ValueType: string; \
    ValueData: "*\shell\SmartImageArchiver\SubCommands"; \

; サブコマンド：Optimize
Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver\SubCommands\Shell\Optimize"; \
    ValueName: "MUIVerb"; ValueType: string; \
    ValueData: "最適化"

Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver\SubCommands\Shell\Optimize\command"; \
    ValueType: string; \
    ValueData: """{app}\SmartImageArchiver.exe"" ""%1"""

Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver\SubCommands\Shell\OptimizeAVIF"; \
    ValueName: "MUIVerb"; ValueType: string; \
    ValueData: "AVIFで最適化"

Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver\SubCommands\Shell\OptimizeAVIF\command"; \
    ValueType: string; \
    ValueData: """{app}\SmartImageArchiver.exe"" ""%1"" --format avif"

Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver\SubCommands\Shell\OptimizeWEBP"; \
    ValueName: "MUIVerb"; ValueType: string; \
    ValueData: "WEBPで最適化"

Root: HKCU; Subkey: "Software\Classes\*\shell\SmartImageArchiver\SubCommands\Shell\OptimizeWEBP\command"; \
    ValueType: string; \
    ValueData: """{app}\SmartImageArchiver.exe"" ""%1"" --format webp"

