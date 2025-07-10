# shiny_app/app.R

# Load and run split components
source("global.R")
source("ui.R")
source("server.R")

# Run the app
shinyApp(ui = ui, server = server)