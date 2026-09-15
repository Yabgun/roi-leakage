# Kuyruk 8 (15 Eyl 2026): KUYRUK5'in adim 3-4'u ve KUYRUK6 gunluk dosyasi cakismasiyla dustu: Run'a verilen cikti adi
# betigin kendi gunluguyle ayniydi (fovea_cost.log vb.), Out-File dosyayi kilitledi, betik ilk log() cagrisinda
# PermissionError verdi. Duzeltilmis, cakismayan cikti adlariyla hemen baslar; makine bosken:
# 1) Adim 3: sifreli dogruluk eslesmesi ve maliyet (~1 saat)
# 2) Adim 4: sizinti denetimi (~40 dk)
# 3) Adim 5: sifreli ozet (HETAL tarzi; maliyet olcumu bos makinede) ve bozuk baglam saldirisi (~2 saat, GPU)
# Kural: Run'a verilen gunluk adi betiklerin kendi gunluklerinden farkli olmali (burada hepsi kuyruk8_ onekli).
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_fovea_core34.ps1
# Ilerleme: results\logs\kuyruk.log (KUYRUK8 satirlari), bitis satiri "KUYRUK8 TAMAMLANDI"; KUYRUK7 bunu bekler.
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

Note "KUYRUK8 BASLADI (run_fovea_core34.ps1)"

[void](Run "experiments.fovea_cost" @("--n-match", "120", "--reps", "3") "kuyruk8_fovea_cost" "KUYRUK8 adim 3: sifreli dogruluk ve maliyet")
[void](Run "experiments.fovea_leakage" @() "kuyruk8_fovea_leakage" "KUYRUK8 adim 4: sizinti denetimi")
[void](Run "experiments.fovea_baselines" @() "kuyruk8_fovea_baselines" "KUYRUK8 adim 5: sifreli ozet (HETAL tarzi)")
[void](Run "experiments.attack_noisy_context" @() "kuyruk8_noisy_context" "KUYRUK8 adim 5: bozuk baglam saldirisi")

Note "KUYRUK8 TAMAMLANDI"
