# GPU kuyrugu 2 (15 Eyl 2026): kesilen tohum tekrarlarinin devami + duzeltilmis degerlendirmeyle tutarlilik tekrarlari.
# Onceki kuyruk (run_queue.ps1) 14 Eyl 19:08'de oturum kapaninca durdu. Beyin tam/baglam tohum 0-4 bitmisti;
# satirlari experiments.recover_from_preds ile CSV'ye eklendi. Bu kuyruk kalanlari calistirir.
# Transfer ve kok neden, degerlendirme duzeltmesinden (fp32 + karisik sira) once calismisti; yeniden calisir.
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_seeds_resume.ps1
# Ilerleme: results\logs\kuyruk.log (BASLADI/BITTI satirlari), bitis satiri "KUYRUK2 TAMAMLANDI".
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

Note "KUYRUK2 BASLADI (run_seeds_resume.ps1)"

# 1) Beyin MR: baglam + 40 piksel, tohum 0-4 (~25 dk)
[void](Run "experiments.attack_context" @("--dataset", "brain", "--views", "baglam_genis40", "--seeds", "0", "1", "2", "3", "4") "attack_context_tohum_beyin40" "tohum: beyin baglam_genis40 s0-4")

# 2) COVID-QU-Ex: tam, baglam, baglam + 40 piksel, tohum 0-4 (~55 dk)
[void](Run "experiments.attack_context" @("--dataset", "covidqu", "--views", "tam", "baglam", "baglam_genis40", "--seeds", "0", "1", "2", "3", "4") "attack_context_tohum_cxr" "tohum: covidqu tam baglam baglam_genis40 s0-4")

# 3) Diger gorusler, tohum 0, duzeltilmis degerlendirmeyle (tablodaki tum satirlar ayni degerlendirmeyle olsun) (~26 dk)
[void](Run "experiments.attack_context" @("--dataset", "brain", "covidqu", "--views", "yalniz_roi", "baglam_genis10", "baglam_genis20", "kutu", "--seeds", "0") "attack_context_s0_duzeltilmis" "tohum0: diger gorusler (duzeltilmis degerlendirme)")

# 4) Gercekcilik testi ve kok neden, duzeltilmis degerlendirmeyle (~30 dk)
[void](Run "experiments.attack_context_transfer" @() "transfer_duzeltilmis" "transfer (duzeltilmis degerlendirme)")
[void](Run "experiments.root_cause" @() "root_cause_duzeltilmis" "kok neden (duzeltilmis degerlendirme)")

[void](Run "experiments.summarize" @() "summarize")
Note "KUYRUK2 TAMAMLANDI"
