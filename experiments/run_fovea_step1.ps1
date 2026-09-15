# GPU kuyrugu 3 (15 Eyl 2026): Cozum Adim 1, odakli temsil bilgi koruma on deneyi (tohum 0, tam izgara).
# Her veri kumesi ayri surecte calisir; experiments.fovea_info biten satirlari CSV'den okuyup atlar, yani
# kuyruk yarida kalirsa ayni betik yeniden baslatilabilir.
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_fovea_step1.ps1
# Ilerleme: results\logs\kuyruk.log (BASLADI/BITTI satirlari), bitis satiri "KUYRUK3 TAMAMLANDI".
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

Note "KUYRUK3 BASLADI (run_fovea_step1.ps1)"

# 1) Beyin MR: 27 yapilandirma x 5 kat x 12 epoch (~1.5-2.5 saat)
[void](Run "experiments.fovea_info" @("--dataset", "brain") "fovea_info_beyin" "cozum adim 1: beyin, tohum 0")

# 2) COVID-QU-Ex: 27 yapilandirma x 5 epoch (~1.3 saat)
[void](Run "experiments.fovea_info" @("--dataset", "covidqu") "fovea_info_cxr" "cozum adim 1: covidqu, tohum 0")

Note "KUYRUK3 TAMAMLANDI"
