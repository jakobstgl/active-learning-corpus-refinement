load_pretrained <- function(path){
    not_all_na <- function(x) any(!is.na(x))
    fasttext <-  data.table::setDT(readr::read_delim(path,
                                                     delim = " ",
                                                     quote = "",
                                                     skip = 1,
                                                     col_names = F,
                                                     col_types = readr::cols())) |>
        dplyr::select(tidyselect::where(not_all_na)) # remove last column which is all NA
    word_vectors <-  as.matrix(fasttext, rownames = 1)
    colnames(word_vectors) = NULL
    rm(fasttext)
    
    return(word_vectors)
}

compute_dimension <- function(terms, pretrained = pretrained, negative_pole, positive_pole){
    
    terms_neg <- terms |> dplyr::filter(pole == negative_pole)
    terms_pos <- terms |> dplyr::filter(pole == positive_pole)
    
    neg_embedd <- pretrained[rownames(pretrained) %in% terms_neg[["term"]],, drop = FALSE]
    pos_embedd <- pretrained[rownames(pretrained) %in% terms_pos[["term"]],, drop = FALSE]
    
    neg_av_nm <- matrix(colMeans(neg_embedd), ncol = ncol(neg_embedd)) |> text2vec::normalize(norm = "l2")
    pos_av_nm <- matrix(colMeans(pos_embedd), ncol = ncol(pos_embedd)) |> text2vec::normalize(norm = "l2")
    
    dim <- (neg_av_nm - pos_av_nm)
    
    return(dim)
    
}


retrieve_embeddings <- function(toks_feats, pattern, window = 10, pretrained = pretrained, group_by, transform_matrix){
    
    dem_pattern <- vector(mode = "list", length = length(pattern))
    dem_grouped <- vector(mode = "list", length = length(pattern))
    names(dem_grouped) <- pattern
    
    for (i in (1:length(pattern))){
        
        dem_pattern[[i]] <- conText::tokens_context(x = toks_feats, pattern = pattern[[i]], window = window) |>
            quanteda::dfm() |>
            conText::dem(pre_trained = pretrained, transform = TRUE, transform_matrix = transform_matrix, verbose = TRUE)
        
        dem_grouped[[i]] <- conText::dem_group(x = dem_pattern[[i]], groups = dem_pattern[[i]]@docvars[[group_by]])
        
    }
    
    return(dem_grouped)
}

compute_allwordsims <- function(tokens_object, dimension, pretrained, group_by){
    
    # Compute tokens subset and retrieve features per Period ------------------
    allwordsims <- function(toks_subset, dimension, pretrained){
        word_vectors <- pretrained[rownames(pretrained) %in% toks_subset,, drop = F]
        allwordsims_dim <- text2vec::sim2(dimension, word_vectors, method = "cosine", norm = "l2")
    }
    
    if (missing(group_by)){
        toks_dfm <- tokens_object |>
            quanteda::dfm() |>
            quanteda::featnames()
        
        return(alwordsims_dim <- allwordsims(toks_dfm, dimension, pretrained))
    }
    
    else {
        grouping_variable <- quanteda::docvars(tokens_object, group_by)
        grouping_variable_length <- length(unique(grouping_variable))
        toks_dfm <- vector(mode = "list", length = grouping_variable_length)
        for (i in (1:grouping_variable_length)){
            toks_dfm[[i]] <- quanteda::tokens_subset(tokens_object, grouping_variable == unique(grouping_variable)[i]) |>
                quanteda::dfm() |>
                quanteda::featnames()
        }
        
        allwordsims_dim <- purrr::map(toks_dfm, \(x) allwordsims(x, dimension = dimension, pretrained = pretrained))
        return(allwordsims_dim)
    }
}

write_scores_doc <- function(embeddings, dimension, tokens_object, pretrained, document_id){
    
    scores <- vector(mode = "list", length = length(embeddings))
    names(scores) <- names(embeddings)
    embeddings <- purrr::map(embeddings, as.matrix)
    
    allwordsims <- compute_allwordsims(tokens_object, dimension, pretrained)
    
    for (j in 1:length(embeddings)){
        vec_grouping <- c()
        groupvar <- rownames(embeddings[[j]]) # document_ids to loop over for the current disease
        
        for (i in 1:length(groupvar)){
            vec_grouping[[i]] <- (text2vec::sim2(embeddings[[j]][i,, drop = F], dimension, method = "cosine", norm = "l2") - mean(allwordsims)) / stats::sd(allwordsims)
        }
        
        scores[[j]] <- tibble::tibble(
            document_id = groupvar,
            stigma_score = vec_grouping
        )
    }
    
    return(scores)
}
