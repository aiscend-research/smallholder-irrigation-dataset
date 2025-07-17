# shiny_app/ui.R
# UI side logic of the app

# This script defines a simplified version of the UI layout for the Zambia Irrigation Explorer Shiny app.
# For now, it includes only a single tab for the interactive map, using shinydashboard to allow expansion later.

library(shiny)
library(shinydashboard)
library(shinyjs)
library(leaflet)

ui <- dashboardPage(
  skin = "blue",
  
  ##### --- Header --- #####
  dashboardHeader(
    title = tags$div(
      style = "display: flex; align-items: center; white-space: normal; line-height: 1.2;",
      
      tags$img(
        src = "logo.png",  # ✅ your Zambia outline image
        height = "40px",
        style = "margin-right: 10px;"
      ),
      
      tags$div(
        HTML("Irrigation Across<br>Zambia"),
        style = "font-size: 20px; font-weight: bold;"
      )
    ),
    titleWidth = 250  # ✅ Expand header width to fit the full text
  )
  
  ,
  
  ##### --- Sidebar Navigation Menu --- #####
  dashboardSidebar(
    width = 250,  # ✅ Set a fixed width for the sidebar
    sidebarMenu(
      id = "tabs",
      
      # Map Viewer Tab shows up first
      menuItem("Map Viewer", tabName = "map", icon = icon("map")),
      
      # Time Series Tab shows up second
      menuItem("Coverage Time Series", tabName = "timeseries", icon = icon("chart-line")),
      
      # Context tab is below
      menuItem("About the Data", tabName = "context", icon = icon("book"))
    )
  ),
  
  ##### --- Main Body Content --- #####
  dashboardBody(
    useShinyjs(),
    tabItems(
      
      ## Map Viewer Tab ##
      tabItem(tabName = "map",
              fluidRow(
               
                 # Show filters and data tables
                column(3,
                       h3("Filter Images"),
                       
                       # Add in the year slider
                       sliderInput("year_range_filter", "Year Range:",
                                   min = 2016, max = 2025,
                                   value = c(2016, 2025), sep = ""),
                       
                       # Add in the certainty slider
                       sliderInput("certainty_filter", "Min Certainty Score:", min = 1, max = 5, value = 4),
                       
                       # Add the water source selection
                       selectInput("water_source_filter", "Water Source?",
                                   choices = c("All", "TRUE", "FALSE"), selected = "All"),
                      
                       # Add option to toggle off unirrigated plots
                        checkboxInput(
                         inputId = "show_zero_coverage",
                         label = "Show Unirrigated Sites",
                         value = TRUE
                       ),
                       
                        br(),
                       
                       # Header for point table info
                       h3("Selected Site"),
                       uiOutput("site_table")
                ),
                
                # Show the map on the side of the filters 
                column(9,
                       leafletOutput("irrigation_map", height = 600)
                )
              )
      ),
      
     ## Time Series Tab ##
     tabItem(tabName = "timeseries",
        fluidPage(
              sidebarLayout(
                sidebarPanel(
                  selectInput("province_filter", "Select Province:",
                              choices = c("All Provinces"),
                              selected = "All Provinces",
                              multiple = FALSE)
                ),
                mainPanel(
                  plotOutput("coverage_time_series_plot", height = "500px"),
                  br(),
                  p("Shows monthly average percent coverage of high-certainty irrigation (certainty ≥ 3), with 95% confidence intervals.")
                )
              )
     )
     ),
     
     ## Context Tab ##
      tabItem(tabName = "context",
              uiOutput("context_html")
      )
      
    )
  )
)