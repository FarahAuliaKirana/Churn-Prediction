# Render r/ab_test_design.Rmd menjadi HTML mandiri (gambar ter-embed).
# Tidak membutuhkan pandoc: knitr -> markdown -> HTML.
#
# Usage (dari root project):
#   Rscript r/render.R
#
# Package: survival (bawaan R), pwr, knitr, markdown
#   install.packages(c("pwr", "knitr", "markdown"))

user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib) && dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))

for (p in c("survival", "pwr", "knitr", "markdown")) {
  if (!requireNamespace(p, quietly = TRUE)) stop("Package belum terpasang: ", p)
}

old <- setwd("r")
on.exit(setwd(old))

knitr::knit("ab_test_design.Rmd", output = "ab_test_design.md", quiet = TRUE)
markdown::mark_html("ab_test_design.md", output = "ab_test_design.html",
                    options = "+embed_resources")
# ab_test_design.md + figures/ tetap disimpan: GitHub menampilkan Markdown (beserta
# gambarnya) langsung di browser, sedangkan HTML hanya bisa dibuka setelah diunduh.
message("Selesai: r/ab_test_design.md (untuk GitHub) dan r/ab_test_design.html")
