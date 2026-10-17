# ============================================================
# 方案1「会看重点的双目测距模块」论文采集脚本
# 采集优先级：
#   第一优先 —— 老师推荐清单中与方案1相关的论文（按年份从新到旧）
#   第二优先 —— 方案1 的技术支撑文献（2024-2026 为主）
# 数据源：Europe PMC（开放获取全文）、PNAS、arXiv（开放预印本）
# 用法：pwsh -File collect_papers.ps1
# 注意：必须用 pwsh(PowerShell 7) 运行。Windows PowerShell 5.1 会把
#       UTF-8 无 BOM 的脚本按 ANSI 解析，导致中文乱码并报语法错误。
# ============================================================

$ErrorActionPreference = 'Continue'
$root = 'D:\认知基础-大作业项目\方案1'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ---------- 第一优先：老师推荐清单中与方案1相关的论文 ----------
$tier1 = @(
  @{ doi='10.1038/s41467-022-33161-8'; year=2022; cite='Favila, Kuhl & Winawer'; title='Perception and memory have distinct spatial tuning properties in human visual cortex'; venue='Nature Communications'; rel='低｜视觉皮层空间调谐，认知背景' },
  @{ doi='10.1038/ncomms12110';          year=2016; cite='Moutsiana et al.';        title='Cortical idiosyncrasies predict the perception of object size'; venue='Nature Communications'; rel='低｜物体大小知觉，认知背景' },
  @{ doi='10.1126/science.1211667';      year=2012; cite='Nagata et al.';            title='Depth perception from image defocus in a jumping spider'; venue='Science'; rel='高｜跳蛛靠离焦测距，仿生依据' },
  @{ doi='10.1073/pnas.1001699107';      year=2010; cite='Nardini, Bedford & Mareschal'; title='Fusion of visual cues is not mandatory in children'; venue='PNAS'; rel='高｜深度线索融合与权重，对应多线索决策' },
  @{ doi='10.1038/nn.2465';              year=2010; cite='Rossi & Roorda';           title='The relationship between visual resolution and cone spacing in the human fovea'; venue='Nature Neuroscience'; rel='中｜中央凹采样极限，对应只看重点' },
  @{ doi='10.1038/nature06814';          year=2008; cite='Nadler, Angelaki & DeAngelis'; title='A neural representation of depth from motion parallax in macaque visual cortex'; venue='Nature'; rel='高｜运动视差深度通路' },
  @{ doi='10.1038/nrn2131';              year=2007; cite='Parker';                    title='Binocular depth perception and the cerebral cortex'; venue='Nature Reviews Neuroscience'; rel='高｜立体视觉权威综述' },
  @{ doi='10.1126/science.1127889';      year=2006; cite='Fernald';                   title='Casting a genetic light on the evolution of eyes'; venue='Science'; rel='中｜眼的演化，仿生立论' },
  @{ doi='10.1038/nn1461';               year=2005; cite='Welchman et al.';           title='3D shape perception from combined depth cues in human visual cortex'; venue='Nature Neuroscience'; rel='高｜多深度线索融合的神经证据' },
  @{ doi='10.1038/415429a';              year=2002; cite='Ernst & Banks';             title='Humans integrate visual and haptic information in a statistically optimal fashion'; venue='Nature'; rel='中高｜最优融合的经典依据' },
  @{ doi='10.1038/35102562';             year=2001; cite='Ooi, Wu & He';             title='Distance determined by the angular declination below the horizon'; venue='Nature'; rel='中高｜距离判断的多线索加权' },
  @{ doi='10.1038/35051081';             year=2001; cite='Wexler et al.';             title='Self-motion and the perception of stationary objects'; venue='Nature'; rel='中｜自运动参与深度知觉' },
  @{ doi='10.1038/71140';                year=2000; cite='Ernst, Banks & Bulthoff';   title='Touch can change visual slant perception'; venue='Nature Neuroscience'; rel='中高｜线索权重可被训练改变' },
  @{ doi='10.1073/pnas.97.1.500';        year=2000; cite='Kaufman & Kaufman';         title='Explaining the moon illusion'; venue='PNAS'; rel='低｜距离知觉背景材料' },
  @{ doi='10.1038/17383';                year=1999; cite='Roorda & Williams';         title='The arrangement of the three cone classes in the living human eye'; venue='Nature'; rel='中｜非均匀采样，仿生传感器依据' },
  @{ doi='10.1038/29299';                year=1998; cite='DeAngelis, Cumming & Newsome'; title='Cortical area MT and the perception of stereoscopic depth'; venue='Nature'; rel='高｜视差神经编码的因果证据' },
  @{ doi='10.1073/pnas.95.7.4002';       year=1998; cite='Coppola et al.';            title='The distribution of oriented contours in the real world'; venue='PNAS'; rel='中｜自然图像统计，类脑前端依据' },
  @{ doi='10.1038/361719a0';             year=1993; cite='Azzopardi & Cowey';         title='Preferential representation of the fovea in the primary visual cortex'; venue='Nature'; rel='中高｜中央凹超表征＝只看重点的生物学依据' },
  @{ doi='10.1038/302821a0';             year=1983; cite='Rossel';                     title='Binocular stereopsis in an insect'; venue='Nature'; rel='高｜螳螂只在中央小区做立体测距' },
  @{ doi='10.1073/pnas.77.9.5572';       year=1980; cite='Held, Birch & Gwiazda';     title='Stereoacuity of human infants'; venue='PNAS'; rel='高｜立体视锐度与精度量级参照' },
  @{ doi='10.1126/science.1188391';      year=1975; cite='Blake & Hirsch';            title='Deficits in binocular depth perception in cats after alternating monocular deprivation'; venue='Science'; rel='中高｜双目细胞对立体视觉的必要性' }
)

# ---------- 第二优先：方案1 的技术支撑文献（arXiv） ----------
$tier2 = [ordered]@{
  '02_立体匹配置信度与不确定度' = @(
    '2101.00431','2401.12001','2412.18703','2408.02796','2308.00728','2304.00152',
    '2008.06447','1905.07287','2404.06273','2508.01275','2506.00324'
  )
  '03_不确定度驱动的鲁棒与高效匹配' = @(
    '2307.14071','2307.16509','2508.01303','2002.03663','2606.02906'
  )
  '04_选择性预测与风险管理' = @(
    '2410.24029','2407.01032','2405.11533','2508.07556','2402.10665','2509.11520','2009.12840'
  )
  '05_主动视觉与中央凹感知' = @(
    '2606.02565','2602.03766','2603.22570','2607.00780','2507.15833','2502.06747','2603.09613'
  )
  '06_自适应计算与anytime感知' = @(
    '1612.02297','2208.12181','2301.13195','1801.00508','2607.05445','2603.06351'
  )
  '07_事件相机立体视觉' = @(
    '2409.17680','2509.18184','2604.02331','2506.06918'
  )
  '08_能效与嵌入式感知' = @(
    '2605.06927','2608.15830','2607.04921'
  )
}

function Get-SafeName([string]$s, [int]$max = 90) {
  $t = ($s -replace '[\\/:*?"<>|]','') -replace '\s+','_'
  $t = $t.Trim('_')
  if ($t.Length -gt $max) { $t = $t.Substring(0, $max) }
  return $t
}

# ---------- 建目录 ----------
$dir1 = '01_老师推荐清单_立体与深度感知'
New-Item -ItemType Directory -Force -Path (Join-Path $root $dir1) | Out-Null
foreach ($k in $tier2.Keys) {
  New-Item -ItemType Directory -Force -Path (Join-Path $root $k) | Out-Null
}
New-Item -ItemType Directory -Force -Path (Join-Path $root '_metadata') | Out-Null

$rows = @()

# ============ 第一优先 ============
Write-Output '===== 第一优先：老师推荐清单 ====='
foreach ($p in ($tier1 | Sort-Object { $_.year } -Descending)) {
  $doi = $p.doi
  $name = "{0}_{1}_{2}" -f $p.year, (Get-SafeName $p.cite 30), (Get-SafeName $p.title)
  $dest = Join-Path (Join-Path $root $dir1) "$name.pdf"
  $status = ''; $src = ''

  if (Test-Path $dest) {
    $status = 'exists'; $src = 'local'
  } else {
    try {
      $q = [uri]::EscapeDataString('DOI:"' + $doi + '"')
      $u = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=$q&resultType=core&format=json&pageSize=1"
      $r = Invoke-RestMethod -Uri $u -TimeoutSec 45
      $hit = $r.resultList.result | Select-Object -First 1
      if ($hit -and $hit.pmcid -and $hit.isOpenAccess -eq 'Y') {
        Invoke-WebRequest -Uri "https://europepmc.org/articles/$($hit.pmcid)?pdf=render" -OutFile $dest -UseBasicParsing -TimeoutSec 120
        $status = 'downloaded'; $src = "EuropePMC $($hit.pmcid)"
      }
    } catch { }
    Start-Sleep -Milliseconds 800

    if ($status -eq '' -and $doi -like '10.1073/*') {
      try {
        Invoke-WebRequest -Uri "https://www.pnas.org/doi/pdf/$doi" -OutFile $dest -UseBasicParsing -TimeoutSec 120
        $status = 'downloaded'; $src = 'PNAS'
      } catch { }
      Start-Sleep -Milliseconds 800
    }

    if ($status -eq '') {
      $status = '需校园网/图书馆获取'
      $src = "https://doi.org/$doi"
      if (Test-Path $dest) { Remove-Item $dest -Force }
    }
  }

  Write-Output ("  [{0}] {1} {2}" -f $status, $p.year, $p.title)
  $rows += [pscustomobject]@{
    优先级 = '1_老师推荐'; 目录 = $dir1; 年份 = $p.year; 作者 = $p.cite; 标题 = $p.title
    出处 = $p.venue; DOI = $doi; 相关度 = $p.rel; 获取状态 = $status; 来源 = $src
  }
  Start-Sleep -Milliseconds 400
}

# ============ 第二优先 ============
Write-Output '===== 第二优先：技术支撑文献 ====='
$allIds = @()
foreach ($k in $tier2.Keys) { $allIds += $tier2[$k] }

$meta = @{}
for ($i = 0; $i -lt $allIds.Count; $i += 20) {
  $batch = $allIds[$i..([Math]::Min($i + 19, $allIds.Count - 1))]
  $u = 'http://export.arxiv.org/api/query?id_list=' + ($batch -join ',') + '&max_results=50'
  try {
    $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 60
    $xml = [xml]$r.Content
    foreach ($e in $xml.feed.entry) {
      $id = (($e.id -split '/abs/')[-1]) -replace 'v\d+$',''
      $meta[$id] = [pscustomobject]@{
        id = $id; published = $e.published.Substring(0,10)
        title = ($e.title -replace '\s+',' ').Trim()
        authors = (($e.author | ForEach-Object { $_.name }) -join ', ')
      }
    }
  } catch { Write-Output "  元数据失败: $($_.Exception.Message)" }
  Start-Sleep -Milliseconds 3200
}

foreach ($k in $tier2.Keys) {
  foreach ($id in $tier2[$k]) {
    $m = $meta[$id]
    if (-not $m) { Write-Output "  [无元数据] $id"; continue }
    $dest = Join-Path (Join-Path $root $k) ("{0}_{1}_{2}.pdf" -f $m.published, $m.id, (Get-SafeName $m.title))
    $status = 'downloaded'
    if (Test-Path $dest) { $status = 'exists' }
    else {
      try { Invoke-WebRequest -Uri "https://arxiv.org/pdf/$($m.id)" -OutFile $dest -UseBasicParsing -TimeoutSec 150 }
      catch { $status = 'failed'; Write-Output "  [失败] $($m.id) $($_.Exception.Message)" }
      Start-Sleep -Seconds 3
    }
    Write-Output ("  [{0}] {1} {2}" -f $status, $m.published, $m.title)
    $rows += [pscustomobject]@{
      优先级 = '2_技术支撑'; 目录 = $k; 年份 = $m.published.Substring(0,4); 作者 = $m.authors
      标题 = $m.title; 出处 = "arXiv:$($m.id)"; DOI = ''; 相关度 = ''; 获取状态 = $status; 来源 = "https://arxiv.org/abs/$($m.id)"
    }
  }
}

$rows | Export-Csv -Path (Join-Path $root '_metadata\采集清单.csv') -NoTypeInformation -Encoding UTF8
$rows | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $root '_metadata\采集清单.json') -Encoding UTF8

$got = ($rows | Where-Object { $_.获取状态 -in @('downloaded','exists') }).Count
Write-Output "===== 完成：共 $($rows.Count) 条，已获取全文 $got 篇 ====="
