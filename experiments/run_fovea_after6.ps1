# GPU kuyrugu 7 (15 Eyl 2026): KUYRUK6 bitince kendiliginden baslar (kuyruk.log'da satir sonu "KUYRUK6 TAMAMLANDI").
# Adim 1 karari sonrasi (secilen: F32_G16 ekonomik, F64_G32 dogru):
# 1) Adim 1, tohum 1-4: beyin (tam, tam_pencere, secilenler, es butceli U32 ve U64, U64_pencere), CXR (tam, secilenler,
#    U32, U64; CXR'de pencere etkisi tohum 0'da sifir) (~2.5 saat) ve sekil
# 2) Adim 2, tohum 1-4: ana satirlar (tam goruntu = Pi_ROI dogrulugu, secilenler, U64), D, D2, C
# 3) Ozgunluk bonusu: butce duyarli odak (tohum 0; Model C ve D)
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_fovea_after6.ps1
# Ilerleme: results\logs\kuyruk.log (KUYRUK7 satirlari), bitis satiri "KUYRUK7 TAMAMLANDI".
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

# Bekleme notu aranan ifadeyi icermez; desen satir sonuna sabit. KUYRUK6 gunluk cakismasiyla durduruldu; onun isleri ve
# KUYRUK5'in adim 3-4'u KUYRUK8'de (run_fovea_core34.ps1), bu kuyruk onun bitisini bekler.
Note "KUYRUK7 BEKLIYOR: kuyruk 8 bitisi (run_fovea_after6.ps1)"
while (-not (Select-String -Path $log -Pattern '\] KUYRUK8 TAMAMLANDI\s*$' -Quiet)) { Start-Sleep -Seconds 60 }
Note "KUYRUK7 BASLADI"

# 1) Adim 1: 5 tohum
# Beyinde ROI bilgisi kazancin buyuk kismini acikliyor (tam_pencere 0.9921 / tam 0.9825); odaklamanin kendi katkisini
# ayirmak icin es butceli es ornekli + ROI penceresi referanslari da 5 tohum. Tohum 0 olan satirlar atlanir.
[void](Run "experiments.fovea_info" @("--dataset", "brain", "--configs", "tam", "tam_pencere", "F32_G16", "F64_G32", "U32", "U32_pencere", "U64", "U64_pencere", "U90", "U90_pencere", "--seeds", "0", "1", "2", "3", "4") "fovea_info_tohum_beyin" "KUYRUK7 adim 1: beyin tohum 0-4")
[void](Run "experiments.fovea_info" @("--dataset", "covidqu", "--configs", "tam", "F32_G16", "F64_G32", "U32", "U64", "--seeds", "1", "2", "3", "4") "fovea_info_tohum_cxr" "KUYRUK7 adim 1: covidqu tohum 1-4")
[void](Run "experiments.fovea_info" @("--plot-only") "fovea_info_sekil_tohum" "KUYRUK7 adim 1: sekil")

# 2) Adim 2: ana satirlarda 5 tohum
[void](Run "experiments.fovea_models" @("--dataset", "brain", "--configs", "U512", "F32_G16", "F64_G32", "U64", "--seeds", "1", "2", "3", "4", "--device", "cuda") "fovea_models_tohum_beyin" "KUYRUK7 adim 2: beyin tohum 1-4")
[void](Run "experiments.fovea_models" @("--dataset", "covidqu", "--configs", "U256", "F32_G16", "F64_G32", "U64", "--seeds", "1", "2", "3", "4", "--device", "cuda") "fovea_models_tohum_cxr" "KUYRUK7 adim 2: covidqu tohum 1-4")

# 3) Ozgunluk bonusu: butce duyarli odak
[void](Run "experiments.fovea_budget" @() "fovea_budget" "KUYRUK7 bonus: butce duyarli odak")

Note "KUYRUK7 TAMAMLANDI"
