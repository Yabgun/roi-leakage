# CPU kuyrugu 1 (15 Eyl 2026): Cozum Adim 2, sifreli calisabilen modeller (D, D2), sifresiz egitim, tohum 0.
# GPU kuyrugu 3 (Adim 1) ile ayni anda calisir; modeller kucuk oldugu icin CPU'da (6 is parcacigi) egitilir.
# experiments.fovea_models biten satirlari CSV'den okuyup atlar; yarida kalirsa ayni betik yeniden baslatilir.
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_fovea_step2_cpu.ps1
# Ilerleme: results\logs\kuyruk.log (CPU1 satirlari), bitis satiri "KUYRUK-CPU1 TAMAMLANDI".
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

Note "KUYRUK-CPU1 BASLADI (run_fovea_step2_cpu.ps1)"

# 1) Beyin MR: 8 temsil x (D, D2) x 5 kat x 3 agirlik azaltma (~1 saat)
[void](Run "experiments.fovea_models" @("--dataset", "brain", "--device", "cpu", "--threads", "6") "fovea_models_beyin" "CPU1 cozum adim 2: beyin, tohum 0")

# 2) COVID-QU-Ex: 8 temsil x (D, D2) x 3 agirlik azaltma (~1-1.5 saat)
[void](Run "experiments.fovea_models" @("--dataset", "covidqu", "--device", "cpu", "--threads", "6") "fovea_models_cxr" "CPU1 cozum adim 2: covidqu, tohum 0")

Note "KUYRUK-CPU1 TAMAMLANDI"
