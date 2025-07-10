# shiny_app/server.R
# Server-side logic of the app

# This server script renders a single interactive leaflet map
# displaying irrigation points filtered by user inputs. Clicking a point
# shows metadata like ID and certainty score.

##### --- Load Libraries and Data --- #####
library(tidyverse)
library(sf)
library(leaflet)
library(viridisLite)


# Load the cleaned map dataset
summary_data_clean <- read_csv("shiny_data/cleaned_shiny_map_data.csv")

# Load the cleaned time series dataset
time_clean <- read_csv("shiny_data/cleaned_shiny_timeseries_data.csv")

##### --- Define Server Logic --- #####
server <- function(input, output, session) {
  
################################################################
  ## MAP TAB ##
  
  ##### --- Reactive Data Filter --- #####
  filtered_data <- reactive({
    df <- summary_data_clean 
    
    # Filter based on toggle
    if (!input$show_zero_coverage) {
      df <- df |> filter(avg_percent_coverage > 0)
    }
    
    # Filter water source status if selected
    if (input$water_source_filter != "All") {
      df <- df |> filter(as.character(water_source) == input$water_source_filter)
    }
    
    # Filter by year and certainty
    df |> filter(
      year >= input$year_range_filter[1],
      year <= input$year_range_filter[2],
      avg_certainty >= input$certainty_filter
    )
  })
  
  ##### --- Render Leaflet Map --- #####
  output$irrigation_map <- renderLeaflet({
   
     # Define color palette
    pal <- colorNumeric(
      palette = "viridis",
      domain = summary_data$log_coverage
    )
    
    # Create the leaflet map with filtered data
    leaflet(data = filtered_data()) |> 
      addProviderTiles(providers$CartoDB.Positron) |> 
      addCircleMarkers(
        # Use coordinates from the data
        lng = ~x,
        lat = ~y,
        
        # Adjust the aesthetics of each point
        color = ~ifelse(avg_percent_coverage == 0, "dimgray", pal(log_coverage)),
        radius = ~sqrt(images) * 2.5,
        stroke = FALSE, fillOpacity = 0.85,
        
        # Add popups with site ID and certainty score
        layerId = ~location_num
      ) |>
      
      # Add in a legend
    addLegend(
      pal = pal,
      values = filtered_data()$log_coverage,
      title = "Coverage (%)",
      labFormat = function(type, cuts, p) {
        paste0(round(expm1(cuts), 1), "%")
      }
    )
    
  })
  
  ##### --- Reactive: Store Selected Point --- #####
  selected_point <- reactiveVal(NULL)
  
  # Add in clicking functionality
  observeEvent(input$irrigation_map_marker_click, {
    click <- input$irrigation_map_marker_click
    if (!is.null(click$id)) {
      point_info <- filtered_data() |> filter(location_num == click$id)
      selected_point(point_info)
    }
  })
  
  ##### --- Output: Render Clicked Point Info Table --- #####
  output$site_table <- renderUI({
    req(selected_point())
    info <- selected_point()
    
    # Display information from the data on the table
    # Render the information table safely
    tags$table(class = "table table-sm",
               tags$tbody(
                 tags$tr(tags$th("District"),
                         tags$td(paste(info$district[[1]]))),
                 tags$tr(tags$th("Province"),
                         tags$td(paste(info$province[[1]]))),
                 tags$tr(tags$th("Number of Images"),
                         tags$td(paste(info$images[[1]]))),
                 tags$tr(tags$th("Average Certainty"),
                         tags$td(round(as.numeric(info$avg_certainty[[1]]), 2))),
                 tags$tr(tags$th("Water Source"),
                         tags$td(as.character(info$water_source_mode[[1]]))),
                 tags$tr(tags$th("Coverage"),
                         tags$td(sprintf("%.3f%%", as.numeric(info$avg_percent_coverage[[1]])))),
                 tags$tr(tags$th("High Certainty Coverage"),
                         tags$td(sprintf("%.3f%%", as.numeric(info$avg_percent_coverage_high[[1]]))))
               )
    )
    
  })
################################################################
  ## TIME SERIES TAB ##
  ##### --- Reactive Data Filter --- #####
  observe({
    updateSelectInput(
      inputId = "province_filter",
      choices = c("All Provinces", sort(unique(join_clean$province))),
      selected = "All Provinces"
    )
  })
  
  ##### --- Render Time Series Plot --- #####
  output$coverage_time_series_plot <- renderPlot({
    req(input$province_filter)
    
    # Filter to selected province and high certainty labels
    time_df <- time_clean |>
      filter(
        (input$province_filter == "All Provinces" | province == input$province_filter),
        irrigation >= 3
      ) |>
      mutate(year_month = as.Date(paste(year, month, day, sep = "-")))
    
    # Summarize coverage
    time_summary_df <- time_df |>
      group_by(year) |>
      summarise(
        mean_coverage = mean(percent_coverage_high_certainty, na.rm = TRUE),
        se = sd(percent_coverage_high_certainty, na.rm = TRUE) / sqrt(n()),
        .groups = "drop"
      ) %>%
      mutate(
        lower = mean_coverage - 1.96 * se,
        upper = mean_coverage + 1.96 * se
      )
    
    # Plot
    ggplot(time_summary_df, aes(x = year, y = mean_coverage)) +
      geom_line(color = "darkgreen", size = 1) +
      geom_point(color = "darkgreen", size = 2) +  # optional: show year points
      geom_ribbon(aes(ymin = lower, ymax = upper), fill = "palegreen", alpha = 0.4) +
      labs(
        x = "Year",
        y = "Avg. % High Certainty Coverage",
        title = paste("Annual High Certainty Irrigation Coverage in", input$province_filter),
        caption = "Shaded area = 95% confidence interval"
      ) +
      theme_minimal() +
      scale_x_continuous(breaks = unique(time_summary_df$year)) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
  })
  
  
################################################################
  ## CONTEXT TAB ##
  
  ##### --- Output: Render Context qmd --- #####
  output$context_html <- renderUI({
    includeHTML("www/context.html")
  })
  
  
}
