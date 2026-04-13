#### Functions used for preprocessing ####
basic_cleaning <- function(dataset, max_snippet_length=20){
  dataset_cleaned <- dataset |> 
    filter(!str_detect(Text, pattern = "ui-button.*")) |> 
    mutate(date = str_extract(Text, "\\d{2}.\\d{2}.\\d{4}"), 
           title = str_extract(Text, ".*[\r\n]") |> 
             str_remove("\\d{2}.\\d{2}.\\d{4}"),
           source = str_extract(title, "^\\D*\\d*\\/\\D*.\\d*.*?(?=,)"),
           article_id = str_extract(source, "^\\D*\\d*\\/\\D*.\\d*"),
           source = str_remove(source, "^\\D*\\d*\\/\\D*.\\d*"),
           text = str_remove(Text, ".*[\r\n]") |> 
               str_remove("^\\D*\\d*\\/\\D*.\\d*") |> 
               str_remove("\\d*\\Z") |> 
               str_squish(),
           title = str_remove(title, ".*(?=;);") |> 
             str_to_lower() |> 
             str_squish()
    ) |> 
    distinct(text, .keep_all = T) |> 
    filter(str_count(text, '\\w+') > max_snippet_length) |> 
    select(-Text, -article_id) |> 
    mutate(
        date = dmy(date) |> year()
    ) |> 
    filter(!between(date, 1990,1999)) |>
    mutate(period = case_when(
      date %in% 2000:2002 ~ 2002,
      date %in% 2003:2005 ~ 2005,
      date %in% 2006:2008 ~ 2008,
      date %in% 2009:2011 ~ 2011,
      date %in% 2012:2014 ~ 2014,
      date %in% 2015:2017 ~ 2017,
      date %in% 2018:2020 ~ 2020,
      date %in% 2021:2024 ~ 2024)
      )
      
  return(dataset_cleaned)  
} 

create_id_subcorpus <- function(x, idx){
    x |> 
        mutate(
            id_subcorpus = idx
        )
}


filtering_in_loop <- function(data, keywords){
    data <- data |> 
        filter(str_detect(string = text, pattern = keywords))
}


title_frequency <- function(dataset_cleaned){
  title_counts <- dataset_cleaned |> 
    count(title, sort = T) |> 
    filter(n>1)
}

plot_distances <- function(data){
  data |> 
    count(similairity) |> 
    ggplot(aes(x = similairity,  y = n)) +
    geom_col()
}

preprocess_dist <- function(dataset_filtered, distances){
  distances <- distances |> 
    filter(!duplicated(paste0(pmax(left_side, right_side), pmin(left_side, right_side)))) |> 
    round(2) |> 
    left_join(dataset_filtered, by = c("left_side" = "id")) |> 
    left_join(dataset_filtered, by = c("right_side" = "id"), suffix = c("", "_comparison")) |> 
    filter(!similairity == 1.00)
}

distance_matching <- function(dataset, distances, threshold){
  distances <- preprocess_dist(dataset, distances) |> 
    filter(date == date_comparison) |> 
    relocate(text_comparison, .after = text) |> 
    filter(similairity > threshold) |> 
    mutate(
      dup_id = case_when(str_length(text) > str_length(text_comparison) ~ right_side,
                         str_length(text) < str_length(text_comparison) ~ left_side)
    )
  
  dup_ids <- distances |> pull(dup_id)
  
  dataset_dplrm <- dataset |> 
    filter(!id %in% dup_ids)
  
  return(dataset_dplrm)
}

get_wordlist <- function(text, pattern){
  v <- str_match(text, pattern) |> 
    #str_unique() |> 
    as_tibble() |> 
    count(V1) |> 
    filter(is.na(V1) == F)
}


get_dimcount <- function(){
  tibble <- tibble |> 
    mutate(
      matches = str_match() 
    ) |> 
    group_by(period) |> 
    count(matches)
}


