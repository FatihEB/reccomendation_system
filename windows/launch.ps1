$ErrorActionPreference = 'Stop'
function Run($Program, [string[]] $Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE." }
}

try {
    $running = Invoke-RestMethod 'http://127.0.0.1:8765/api/models' -TimeoutSec 2 -ErrorAction Stop
} catch {
    $running = @()
}
if ($running -contains 'taste' -and $running -contains 'collaborative') {
    Start-Process 'http://127.0.0.1:8765/'
    exit 0
}

$root = (Get-Location).Path
$runtime = Join-Path $root 'data\windows-runtime'
$null = New-Item -ItemType Directory -Force -Path $runtime
$uv = Join-Path $runtime 'uv.exe'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
$env:UV_CACHE_DIR = Join-Path $runtime 'cache'
$env:UV_NO_MODIFY_PATH = '1'
$python = Join-Path $root '.venv\Scripts\python.exe'
$setup = Join-Path $runtime 'requirements.sha256'
$requirements = (Get-FileHash -Algorithm SHA256 requirements.txt).Hash
if (!(Test-Path $python) -or !(Test-Path $setup) -or (Get-Content $setup -Raw).Trim() -ne $requirements) {
    if (!(Test-Path $uv)) {
        $archive = Join-Path $runtime 'uv.zip'
        Run curl.exe @('--fail','--location','--retry','2','--output',$archive,
            'https://github.com/astral-sh/uv/releases/download/0.9.7/uv-x86_64-pc-windows-msvc.zip')
        if ((Get-FileHash -Algorithm SHA256 $archive).Hash -ne '5d250c32d3604e28dbe18dc65c668ff628c53e00dde2c642576e831e4a60da64') {
            throw 'uv archive checksum mismatch.'
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $runtime -Force
        if (!(Test-Path $uv)) {
            $downloaded = Get-ChildItem -LiteralPath $runtime -Recurse -Filter uv.exe | Select-Object -First 1
            Copy-Item -LiteralPath $downloaded.FullName -Destination $uv
        }
    }
    if (!(Test-Path $python)) { Run $uv @('venv','--python','3.12.12','--managed-python','.venv') }
    Run $uv @('pip','install','--python',$python,'-r','requirements.txt')
    Set-Content -LiteralPath $setup -Value $requirements
}
if (!(Test-Path 'data\ml-32m.zip') -or !(Test-Path 'data\mooduel-v1.0.csv')) { Run $python @('snapshots.py') }
if (!(Test-Path 'data\movielens-model.npz')) { Run $python @('train.py','--factorizer','windows\factorize.exe') }
if (!(Test-Path 'data\imdb-credits-local.json') -or !(Test-Path 'data\tag-genome-traits.npz')) {
    Run $python @('taste_data.py','taste')
    Run $python @('taste.py')
}
Run $python @('app.py','--open-browser')
