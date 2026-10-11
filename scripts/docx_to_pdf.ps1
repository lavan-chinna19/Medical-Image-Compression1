$ErrorActionPreference = "Stop"

$docxPath = (Resolve-Path "report/report.docx").Path
$pdfPath = [System.IO.Path]::ChangeExtension($docxPath, "pdf")

Write-Host "Opening Word application..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    Write-Host "Opening document: $docxPath"
    $doc = $word.Documents.Open($docxPath)
    
    if (Test-Path $pdfPath) {
        Remove-Item $pdfPath -Force
    }
    
    Write-Host "Exporting to PDF: $pdfPath"
    $doc.ExportAsFixedFormat($pdfPath, 17) # 17 = wdExportFormatPDF
    
    $doc.Close([ref]$false)
    Write-Host "Export finished."
}
catch {
    Write-Error "Exception occurred: $_"
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}

if (Test-Path $pdfPath) {
    $size = (Get-Item $pdfPath).Length
    Write-Host "SUCCESS: PDF created ($size bytes)"
} else {
    Write-Host "FAILURE: PDF not found"
}
