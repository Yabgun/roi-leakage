# GPU kuyrugu 5 (15 Eyl 2026): KUYRUK3 bitince kendiliginden baslar (kuyruk.log'da "KUYRUK3 TAMAMLANDI" satirini bekler).
# 1) Adim 1 beyin ek referanslari (tam_pencere, U64_pencere, U90_pencere; biten satirlar atlanir) ve sekil
# 2) Adim 2 v3 (D, D2, C): beyin, covidqu
# 3) Makine bosken: Adim 3 tam kosu (dogruluk eslesmesi 120 ornek, 3 tekrar maliyet) ve Adim 4 sizinti denetimi
# Her adim ayri surecte; biri hata verirse sonraki yine calisir (cikis kodu kuyruk.log'da).
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_fovea_after3.ps1
# Ilerleme: results\logs\kuyruk.log (KUYRUK5 satirlari), bitis satiri "KUYRUK5 TAMAMLANDI".
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

# Dikkat: bekleme notu aranan ifadeyi icermemeli; desen satir sonuna sabitlidir (ilk surumde kendi notunu bulup hemen basladi).
Note "KUYRUK5 BEKLIYOR: kuyruk 3 bitisi (run_fovea_after3.ps1)"
while (-not (Select-String -Path $log -Pattern '\] KUYRUK3 TAMAMLANDI\s*$' -Quiet)) { Start-Sleep -Seconds 60 }
Note "KUYRUK5 BASLADI"

# 1) Adim 1: beyin ek referanslari ve sekil (~15 dk)
[void](Run "experiments.fovea_info" @("--dataset", "brain") "fovea_info_beyin_ek" "KUYRUK5 adim 1: beyin ek referanslari")
[void](Run "experiments.fovea_info" @("--plot-only") "fovea_info_sekil" "KUYRUK5 adim 1: sekil ve ozet")

# 2) Adim 2 v3: D, D2, C (~1-2 saat)
[void](Run "experiments.fovea_models" @("--dataset", "brain", "--device", "cuda") "fovea_models_beyin_v3" "KUYRUK5 adim 2 v3: beyin")
[void](Run "experiments.fovea_models" @("--dataset", "covidqu", "--device", "cuda") "fovea_models_cxr_v3" "KUYRUK5 adim 2 v3: covidqu")

# 3) Adim 3 ve 4: makine bosken (~2 saat)
# Not (22:20): ilk surumde cikti adlari betiklerin kendi gunlukleriyle ayniydi (fovea_cost.log, fovea_leakage.log);
# PermissionError ile dustuler. Adlar duzeltildi; bu iki is KUYRUK8'de (run_fovea_core34.ps1) yeniden calisti.
[void](Run "experiments.fovea_cost" @("--n-match", "120", "--reps", "3") "kuyruk5_fovea_cost" "KUYRUK5 adim 3: sifreli dogruluk ve maliyet")
[void](Run "experiments.fovea_leakage" @() "kuyruk5_fovea_leakage" "KUYRUK5 adim 4: sizinti denetimi")

Note "KUYRUK5 TAMAMLANDI"
