# GPU kuyrugu 6 (15 Eyl 2026): KUYRUK5 bitince kendiliginden baslar (kuyruk.log'da satir sonu "KUYRUK5 TAMAMLANDI").
# Cozum Adim 5 (rakipler), tohum 0:
# 1) Sifreli ozet (HETAL tarzi): ImageNet ResNet-18 ozeti + sifreli D/D2; maliyet olcumu makine bosken (~15 dk)
# 2) Gurultulu / bulanik baglam saldirisi (Bi-CryptoNets tarzi) ve ayni duzeyde fayda (~1.5-2 saat)
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_fovea_after5.ps1
# Ilerleme: results\logs\kuyruk.log (KUYRUK6 satirlari), bitis satiri "KUYRUK6 TAMAMLANDI".
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

# Bekleme notu aranan ifadeyi icermez; desen satir sonuna sabit.
Note "KUYRUK6 BEKLIYOR: kuyruk 5 bitisi (run_fovea_after5.ps1)"
while (-not (Select-String -Path $log -Pattern '\] KUYRUK5 TAMAMLANDI\s*$' -Quiet)) { Start-Sleep -Seconds 60 }
Note "KUYRUK6 BASLADI"

# Not (22:22): cikti adlari betiklerin kendi gunlukleriyle ayniydi (PermissionError); duzeltildi. Bu kuyruk durduruldu,
# isleri KUYRUK8'de (run_fovea_core34.ps1).
[void](Run "experiments.fovea_baselines" @() "kuyruk6_fovea_baselines" "KUYRUK6 adim 5: sifreli ozet (HETAL tarzi)")
[void](Run "experiments.attack_noisy_context" @() "kuyruk6_noisy_context" "KUYRUK6 adim 5: gurultulu baglam saldirisi")

Note "KUYRUK6 TAMAMLANDI"
