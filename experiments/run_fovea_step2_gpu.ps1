# GPU kuyrugu 4 (15 Eyl 2026): Cozum Adim 2 protokol v2, sifreli calisabilen modeller (D, D2), tohum 0.
# KUYRUK-CPU1 (run_fovea_step2_cpu.ps1) bilerek durduruldu: tam goruntu modeli agirlik azaltma ve epoch sinirina
# dayandi. v2: veri GPU bellegine yuklenir, agirlik azaltma sinirdaysa arama genisler, en cok 200 epoch.
# KUYRUK3 (Adim 1 ResNet izgarasi) ile ayni anda GPU'da calisir; modeller kucuk, bellek toplami 8 GB altinda.
# experiments.fovea_models biten satirlari CSV'den okuyup atlar; yarida kalirsa ayni betik yeniden baslatilir.
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_fovea_step2_gpu.ps1
# Ilerleme: results\logs\kuyruk.log (KUYRUK4 satirlari), bitis satiri "KUYRUK4 TAMAMLANDI".
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".\.venv\Scripts\python.exe"
$log = ".\results\logs\kuyruk.log"
$noise = 'UserWarning|warnings.warn|NativeCommandError|CategoryInfo|FullyQualifiedErrorId|At line:|^\s*\+'

function Note([string]$msg) {
  # Write-Host: fonksiyonun donus degerine karismaz
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

function Run([string]$module, [string[]]$extra, [string]$logName, [string]$label) {
  if (-not $label) { $label = "$module $($extra -join ' ')" }
  Note "BASLADI: $label"
  & $py -m $module @extra 2>&1 | Where-Object { "$_" -notmatch $noise } | Out-File -FilePath ".\results\logs\$logName.log" -Encoding utf8
  $code = [int]$LASTEXITCODE
  Note "BITTI: $label (cikis=$code)"
  return $code
}

Note "KUYRUK4 BASLADI (run_fovea_step2_gpu.ps1)"

# 1) Beyin MR: 9 temsil x (D, D2) x 5 kat, GPU
[void](Run "experiments.fovea_models" @("--dataset", "brain", "--device", "cuda") "fovea_models_beyin_v2" "KUYRUK4 cozum adim 2 v2: beyin, tohum 0")

# 2) COVID-QU-Ex: 9 temsil x (D, D2), GPU
[void](Run "experiments.fovea_models" @("--dataset", "covidqu", "--device", "cuda") "fovea_models_cxr_v2" "KUYRUK4 cozum adim 2 v2: covidqu, tohum 0")

Note "KUYRUK4 TAMAMLANDI"
