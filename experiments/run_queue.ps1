# GPU iş kuyruğu: adımları sırayla çalıştırır.
# Hızlı sürümü (--quick) başarılı olan betiğin tam sürümü çalışır.
# Çalıştırma (proje kökünden): powershell -NoProfile -File experiments\run_queue.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".\.venv\Scripts\python.exe"
$log = ".\results\logs\kuyruk.log"
$noise = 'UserWarning|warnings.warn|NativeCommandError|CategoryInfo|FullyQualifiedErrorId|At line:|^\s*\+'

function Note([string]$msg) {
  # Write-Host kullanılır: fonksiyonun dönüş değerine karışmaz
  $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
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

# 1) Savunma: düzeltilmiş değerlendirmeyle önce hızlı doğrulama (tamamen_gizli satırı 0.5 olmalı), sonra tam sürüm
$quick = Run "experiments.defense_expansion" @("--quick") "defense_hizli"
if ($quick -eq 0) { [void](Run "experiments.defense_expansion" @() "defense") } else { Note "ATLANDI (hizli surum basarisiz): experiments.defense_expansion" }

# 2) Tohum tekrarları: ana görüşler, tohum 0-4, düzeltilmiş (fp32 + karışık sıra) değerlendirmeyle
[void](Run "experiments.attack_context" @("--dataset", "brain", "covidqu", "--views", "tam", "baglam", "baglam_genis40", "--seeds", "0", "1", "2", "3", "4") "attack_context_tohumlar" "tohum tekrarlari")

[void](Run "experiments.summarize" @() "summarize")
Note "KUYRUK TAMAMLANDI"
