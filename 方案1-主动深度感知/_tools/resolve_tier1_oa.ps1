# ============================================================
# 第一优先文献（老师推荐清单）开放获取解析脚本
# 逻辑：用 Unpaywall 查询每篇的合法开放获取版本，逐个尝试下载，
#       校验文件确实是 PDF；PMC 链接在本机被拦截，仅记录待手动访问。
# 用法：pwsh -File resolve_tier1_oa.ps1
# ============================================================

$ErrorActionPreference = 'Continue'
$root = 'D:\认知基础-大作业项目\方案1'
$dir1 = Join-Path $root '01_老师推荐清单_立体与深度感知'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$tier1 = @(
  @{ doi='10.1038/s41467-022-33161-8'; year=2022; cite='Favila_et_al';        title='Perception and memory have distinct spatial tuning properties in human visual cortex'; venue='Nature Communications'; rel='低｜视觉皮层空间调谐，认知背景' },
  @{ doi='10.1038/ncomms12110';          year=2016; cite='Moutsiana_et_al';     title='Cortical idiosyncrasies predict the perception of object size'; venue='Nature Communications'; rel='低｜物体大小知觉，认知背景' },
  @{ doi='10.1126/science.1211667';      year=2012; cite='Nagata_et_al';        title='Depth perception from image defocus in a jumping spider'; venue='Science'; rel='高｜跳蛛靠离焦测距，仿生依据' },
  @{ doi='10.1073/pnas.1001699107';      year=2010; cite='Nardini_et_al';       title='Fusion of visual cues is not mandatory in children'; venue='PNAS'; rel='高｜深度线索融合与权重' },
  @{ doi='10.1038/nn.2465';              year=2010; cite='Rossi_Roorda';        title='The relationship between visual resolution and cone spacing in the human fovea'; venue='Nature Neuroscience'; rel='中｜中央凹采样极限' },
  @{ doi='10.1038/nature06814';          year=2008; cite='Nadler_et_al';        title='A neural representation of depth from motion parallax in macaque visual cortex'; venue='Nature'; rel='高｜运动视差深度通路' },
  @{ doi='10.1038/nrn2131';              year=2007; cite='Parker';              title='Binocular depth perception and the cerebral cortex'; venue='Nature Reviews Neuroscience'; rel='高｜立体视觉权威综述' },
  @{ doi='10.1126/science.1127889';      year=2006; cite='Fernald';             title='Casting a genetic light on the evolution of eyes'; venue='Science'; rel='中｜眼的演化，仿生立论' },
  @{ doi='10.1038/nn1461';               year=2005; cite='Welchman_et_al';      title='3D shape perception from combined depth cues in human visual cortex'; venue='Nature Neuroscience'; rel='高｜多深度线索融合' },
  @{ doi='10.1038/415429a';              year=2002; cite='Ernst_Banks';         title='Humans integrate visual and haptic information in a statistically optimal fashion'; venue='Nature'; rel='中高｜最优融合的经典依据' },
  @{ doi='10.1038/35102562';             year=2001; cite='Ooi_Wu_He';           title='Distance determined by the angular declination below the horizon'; venue='Nature'; rel='中高｜距离判断的多线索加权' },
  @{ doi='10.1038/35051081';             year=2001; cite='Wexler_et_al';        title='Self-motion and the perception of stationary objects'; venue='Nature'; rel='中｜自运动参与深度知觉' },
  @{ doi='10.1038/71140';                year=2000; cite='Ernst_Banks_Bulthoff';title='Touch can change visual slant perception'; venue='Nature Neuroscience'; rel='中高｜线索权重可被训练改变' },
  @{ doi='10.1073/pnas.97.1.500';        year=2000; cite='Kaufman_Kaufman';     title='Explaining the moon illusion'; venue='PNAS'; rel='低｜距离知觉背景材料' },
  @{ doi='10.1038/17383';                year=1999; cite='Roorda_Williams';     title='The arrangement of the three cone classes in the living human eye'; venue='Nature'; rel='中｜非均匀采样' },
  @{ doi='10.1038/29299';                year=1998; cite='DeAngelis_et_al';     title='Cortical area MT and the perception of stereoscopic depth'; venue='Nature'; rel='高｜视差神经编码的因果证据' },
  @{ doi='10.1073/pnas.95.7.4002';       year=1998; cite='Coppola_et_al';       title='The distribution of oriented contours in the real world'; venue='PNAS'; rel='中｜自然图像统计' },
  @{ doi='10.1038/361719a0';             year=1993; cite='Azzopardi_Cowey';     title='Preferential representation of the fovea in the primary visual cortex'; venue='Nature'; rel='中高｜中央凹超表征' },
  @{ doi='10.1038/302821a0';             year=1983; cite='Rossel';              title='Binocular stereopsis in an insect'; venue='Nature'; rel='高｜螳螂中央小区立体测距' },
  @{ doi='10.1073/pnas.77.9.5572';       year=1980; cite='Held_et_al';          title='Stereoacuity of human infants'; venue='PNAS'; rel='高｜立体视锐度与精度量级' },
  @{ doi='10.1126/science.1188391';      year=1975; cite='Blake_Hirsch';        title='Deficits in binocular depth perception in cats after alternating monocular deprivation'; venue='Science'; rel='中高｜双目细胞的必要性' }
)

function Test-Pdf([string]$path) {
  try {
    $b = [System.IO.File]::ReadAllBytes($path)
    if ($b.Length -lt 5000) { return $false }
    return ([char]$b[0] -eq '%' -and [char]$b[1] -eq 'P' -and [char]$b[2] -eq 'D' -and [char]$b[3] -eq 'F')
  } catch { return $false }
}

$rows = @()
Write-Output '===== 解析老师推荐清单的开放获取版本 ====='

foreach ($p in ($tier1 | Sort-Object { $_.year } -Descending)) {
  $doi = $p.doi
  $base = "{0}_{1}_{2}" -f $p.year, $p.cite, (($p.title -replace '[\\/:*?"<>|]','') -replace '\s+','_')
  if ($base.Length -gt 120) { $base = $base.Substring(0,120) }
  $dest = Join-Path $dir1 "$base.pdf"
  $status = ''; $src = ''

  if (Test-Path $dest) {
    $status = 'exists'; $src = 'local'
  } else {
    $urls = @()
    try {
      $r = Invoke-RestMethod -Uri "https://api.unpaywall.org/v2/$doi`?email=student@example.edu" -TimeoutSec 45
      foreach ($loc in @($r.best_oa_location) + @($r.oa_locations)) {
        if (-not $loc) { continue }
        if ($loc.url_for_pdf) { $urls += $loc.url_for_pdf }
        if ($loc.url)         { $urls += $loc.url }
      }
    } catch { Write-Output "  Unpaywall 失败 $doi : $($_.Exception.Message)" }
    $urls = $urls | Where-Object { $_ } | Select-Object -Unique

    $blocked = @()
    foreach ($u in $urls) {
      if ($u -match 'ncbi\.nlm\.nih\.gov|pmc\.ncbi') { $blocked += $u; continue }
      try {
        Invoke-WebRequest -Uri $u -OutFile $dest -UseBasicParsing -TimeoutSec 120 `
          -Headers @{'User-Agent'='Mozilla/5.0 (Windows NT 10.0; Win64; x64)'} -MaximumRedirection 8
        if (Test-Pdf $dest) { $status = 'downloaded'; $src = $u; break }
        else { Remove-Item $dest -Force -ErrorAction SilentlyContinue }
      } catch { }
      Start-Sleep -Milliseconds 700
    }
    if ($status -eq '' -and $blocked.Count -gt 0) {
      $status = 'PMC可免费阅读(需手动)'; $src = ($blocked | Select-Object -First 1)
    }
    if ($status -eq '') { $status = '需校园网/图书馆获取'; $src = "https://doi.org/$doi" }
  }

  Write-Output ("  [{0}] {1} {2}" -f $status, $p.year, $p.title)
  $rows += [pscustomobject]@{
    优先级='1_老师推荐'; 目录='01_老师推荐清单_立体与深度感知'; 年份=$p.year; 作者=$p.cite; 标题=$p.title
    出处=$p.venue; DOI=$doi; 相关度=$p.rel; 获取状态=$status; 来源=$src
  }
  Start-Sleep -Milliseconds 600
}

$rows | Export-Csv -Path (Join-Path $root '_metadata\老师推荐清单_获取情况.csv') -NoTypeInformation -Encoding UTF8
$got = ($rows | Where-Object { $_.获取状态 -in @('downloaded','exists') }).Count
Write-Output "===== 完成：$($rows.Count) 条，拿到全文 $got 篇 ====="
