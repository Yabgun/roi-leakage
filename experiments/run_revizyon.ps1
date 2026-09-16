# GPU kuyrugu 10 (16 Eyl 2026): inceleme raporunun istedigi ek deneyler, ilerleme yuzdesi ve tahmini bitisle.
#   S1: beyin MR normalizasyonu yalnizca gorunur piksellerden (baglam saldirisi, Grad-CAM, savunma, bozuk baglam)
#   S2: U64 + 3 geometri degeri referansi (Model D, D2, C; 5 tohum)
#   S3: Pi_ROI maliyet olcumu en az 3 tekrarla (CPU; GPU isleri bittikten sonra, makine bosken)
#   sonra: savunma hiz kazanci yeniden hesabi, birlesik ozet ve Pareto sekilleri, OZET.md
# Is listesi ve yuzde hesabinin tek kaynagi: experiments\queue_progress.py (--kuyruk 10)
# Calistirma (proje kokunden): powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_revizyon.ps1
# Ilerleme: pencerenin ustundeki cubuk ve pencere basligi (yuzde, kalan sure, tahmini bitis); 10 dakikada bir satir.
# Pencereyi kapatmak kuyrugu durdurur; ayni komutla yeniden baslatilinca biten satirlari atlayarak devam eder.
# Bu pencere acikken bilgisayar uykuya gecmez (ayar degistirmez; pencere kapaninca kendiliginden kalkar).
param([switch]$Deneme)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".\.venv\Scripts\python.exe"
$tag = if ($Deneme) { "kuyruk10_deneme" } else { "kuyruk10" }
$log = if ($Deneme) { ".\results\logs\kuyruk10_deneme.log" } else { ".\results\logs\kuyruk.log" }
$pollSeconds = if ($Deneme) { 2 } else { 30 }
$noteMinutes = if ($Deneme) { 0.05 } else { 10 }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$progressArgs = @("-m", "experiments.queue_progress", "--kuyruk", "10")
if ($Deneme) { $progressArgs += "--deneme" }

function Note([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

function Show([string[]]$f) {
  Write-Progress -Activity "KUYRUK10: inceleme ek deneyleri" -Status $f[1] -PercentComplete ([math]::Max(0, [math]::Min(100, [int]$f[0])))
  try { $Host.UI.RawUI.WindowTitle = $f[2] } catch { }
}

try {
  Add-Type -Namespace TezKuyruk -Name Guc -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
  [void][TezKuyruk.Guc]::SetThreadExecutionState([uint32]2147483649)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
} catch {
  Note "UYARI: uyku engeli kurulamadi: $($_.Exception.Message)"
}

$planOut = & $py @progressArgs --plan
if ($LASTEXITCODE -ne 0) { Note "KUYRUK10 HATA: is plani olusturulamadi (cikis=$LASTEXITCODE)"; exit 1 }
$jobs = @()
$planNote = ""
foreach ($line in $planOut) {
  $f = "$line".Split("`t")
  if ($f[0] -eq "IS") {
    $jobs += [pscustomobject]@{ Idx = [int]$f[1]; Key = $f[2]; Label = $f[3]; Module = $f[4]
                                Args = @($f[5].Split(" ", [StringSplitOptions]::RemoveEmptyEntries))
                                Optional = ($f[6] -eq "1"); AlreadyDone = ($f[7] -eq "1") }
  } elseif ($f[0] -eq "PLAN") {
    $planNote = $f[1]
  }
}
Note "KUYRUK10 BASLADI: $planNote"

$done = New-Object System.Collections.Generic.List[int]
foreach ($j in $jobs) {
  $n = "{0}/{1}" -f ($j.Idx + 1), $jobs.Count
  $script = ".\" + ($j.Module -replace '\.', '\') + ".py"
  if ($j.AlreadyDone) { Note "ATLANDI: is $n $($j.Label) (satirlari zaten var)"; $done.Add($j.Idx); continue }
  if ($j.Optional -and -not (Test-Path $script)) { Note "ATLANDI: is $n $($j.Label) (betik yok: $script)"; $done.Add($j.Idx); continue }
  $out = ".\results\logs\$tag`_$($j.Key).out.log"
  $err = ".\results\logs\$tag`_$($j.Key).err.log"
  Note "BASLADI: is $n $($j.Label)"
  $p = Start-Process -FilePath $py -ArgumentList (@("-m", $j.Module) + $j.Args) -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  $null = $p.Handle  # PowerShell 5.1: cikis kodu ancak tutamac onceden alinirsa okunur
  $lastNote = Get-Date
  do {
    $r = @(& $py @progressArgs --poll --job $j.Idx ("--done=" + ($done -join ",")) --log $out 2>$null)
    if ($LASTEXITCODE -eq 0 -and $r.Count -gt 0) {
      $f = "$($r[-1])".Split("`t")
      if ($f.Count -ge 4) {
        Show $f
        if (((Get-Date) - $lastNote).TotalMinutes -ge $noteMinutes) { Note $f[3]; $lastNote = Get-Date }
      }
    }
  } until ($p.WaitForExit($pollSeconds * 1000))
  $p.WaitForExit()
  $code = $p.ExitCode
  Note "BITTI: is $n $($j.Label) (cikis=$code)"
  if ($code -ne 0) { Note "  hata ayrintisi: $err" }
  $done.Add($j.Idx)
}

Write-Progress -Activity "KUYRUK10: inceleme ek deneyleri" -Completed
try { $Host.UI.RawUI.WindowTitle = "%100 | KUYRUK10 TAMAMLANDI" } catch { }
[void][TezKuyruk.Guc]::SetThreadExecutionState([uint32]2147483648)  # ES_CONTINUOUS: uyku engelini kaldir
Note "KUYRUK10 TAMAMLANDI"
