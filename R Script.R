library(ROAuth)
library(twitteR)
library(dplyr)
library(ggplot2)
library(stringr)
library(wordcloud)
library(tm)
library(RColorBrewer)
library(topicmodels)
library(RTextTools)
library(tidyr)
library(tidytext)
library(NMF)

#authenticating twitter using consumer and access keys
consumer_key <- consumer_key_nt
consumer_secret <- consumer_secret_nt
access_token <- access_token_nt
access_secret <- access_secret_nt

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

#saving keywords
KeyWords <- c("#Felix Tshisekedi", "#Kamerhe", "Kamerhe", "attaque Congo", 
              "Abus RD Congo", "#Kinshasa", "#Kinshasa election", "Joseph Kabila", 
              "militants", "sanctions", "crimes de guerres Congo", "liste des candidats", 
              "#Kabila", "#Katumbi", "electoral commission", "Jean pierre bemba", "#drCongo", 
              "coalision", "Fayulu", "compagne electorale", "23 decembre 2018", "#Felix", 
              "#Shadari", "CENI", "mort", "Morts Congo", "CENI", "Katumbi", "Tshisekedi", 
              "massacres RDCongo", "violence RD Congo", "groupes armés", "attaques armées", 
              "massacre village Congo", "l’accord du 31 décembre", "Les élections présidentielle", 
              "l'insécurité politique Kinshasa", "L'accord de la Saint-Sylvestre", 
              "le mouvement Lutte pour le changement", "Lucha", "listes électorales Congo")

#pulling tweets from twitter
tweets <- searchTwitteR(KeyWords, n = 1000, lang = "en")
tweets.df <- twListToDF(tweets)

#creating a corpus
myCorpus <- Corpus(VectorSource(tweets.df$text)) 
# convert to lower case 
myCorpus <- tm_map(myCorpus, content_transformer(str_to_lower))
# remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x) 
myCorpus <- tm_map(myCorpus, content_transformer(removeURL)) 
# remove anything other than English letters or space 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x) 
myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct)) 
# remove stopwords 
myStopwords <- myStopwords <- c(stopwords('french')) 
myCorpus <- tm_map(myCorpus, removeWords, myStopwords) 
# remove extra whitespace 
myCorpus <- tm_map(myCorpus, stripWhitespace)

# term document matrix 
tdm <- TermDocumentMatrix(myCorpus, control = list(wordLengths = c(1, Inf))) 
tdm

#finding frequent terms
freq.terms <- findFreqTerms(tdm, lowfreq = 40)

#frequent terms counts graphs
term.freq <- rowSums(as.matrix(tdm)) 
term.freq <- subset(term.freq, term.freq >= 40) 
df <- data.frame(term = names(term.freq), freq = term.freq)

#arranging frequent terms by descending order
df <- arrange(df, desc(df$freq))

#plotting top 20 words
ggplot(df[1:20,], aes(x=term, y=freq)) + geom_bar(stat="identity") + xlab("Terms") + ylab("Count") + coord_flip() + theme_classic()

#wordcloud
m <- as.matrix(tdm) 
# calculate the frequency of words and sort it by frequency 
word.freq <- sort(rowSums(m), decreasing = T) 
# colors 
pal <- brewer.pal(9, "BuGn")[-(1:4)]
#plot word cloud 
wordcloud(words = names(word.freq), freq = word.freq, min.freq = 3, random.order = F, colors = pal)


#starting with k-means clustering
tdm_tfidf <- weightTfIdf(tdm)

#removing sparse terms
tdm_tfidf <- removeSparseTerms(tdm_tfidf, 0.999)
tfidf_matrix <- as.matrix(tdm_tfidf)

# transpose the matrix to cluster documents (tweets)
m3 <- t(tfidf_matrix)

#eucledian distance matrix
#dist.matrix = dist(tfidf_matrix, method = "euclidean")

#finding optimal k using elbow method - Compute and plot wss for k = 2 to k = 15.
set.seed(123)
k.max <- 15
data <- m3
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})

#plotting the elbow curve
plot(1:k.max, wss, type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#k-means with k=5
set.seed(123)
clustering.kmeans <- kmeans(m3, 5) 

#cluster size
clustersize <- clustering.kmeans$size
clustersize

#centers
center <- clustering.kmeans$centers
center

#clusters
cluster <- clustering.kmeans$cluster
cluster


#most frequent terms in each cluster
k = 5
for (i in 1:k) {
  cat(paste("cluster ", i, ": ", sep=""))
  s <- sort(clustering.kmeans$centers[i,], decreasing=T)
  cat(names(s)[1:3], "\n")
}


#LDA Modelling
doc.lengths <- rowSums(as.matrix(DocumentTermMatrix(myCorpus)))
dtm <- DocumentTermMatrix(myCorpus[doc.lengths > 0])

#LDA
SEED = sample(1:1000000, 1)
k = 10

#building multiple models
models <- list(
  CTM       = CTM(dtm, k = k, control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3))),
  VEM       = LDA(dtm, k = k, control = list(seed = SEED)),
  VEM_Fixed = LDA(dtm, k = k, control = list(estimate.alpha = FALSE, seed = SEED)),
  Gibbs     = LDA(dtm, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000,
                                                               thin = 100,    iter = 1000))
)


# Top 10 terms of each topic for each model
lda_topic_labels <- lapply(models, terms, 10)


#NMF
rownames(tdm1) <- tdm1[,1]
tdm1[,1] <- NULL

res <- nmf(m,5,"KL")

w <- basis(res)
dim(w)

dfnmf <- as.data.frame(w)
head(dfnmf,10)

dfnmf$total <- rowSums(df)
dfnmf$word<-rownames(df)
colnames(dfnmf) <- c("doc1","doc2","doc3","doc4","doc5", "total","word")
dfnmf <-dfnmf[order(-dfnmf$total),] 
head(dfnmf,20)

wordMatrix = as.data.frame(w)
wordMatrix$word<-rownames(wordMatrix)
colnames(wordMatrix) <- c("doc1","doc2","doc3","doc4","doc5","word")

# Topic 1
newdata <-wordMatrix[order(-wordMatrix$doc1),] 
head(newdata)

d <- newdata
dftopic <- as.data.frame(cbind(d[1:10,]$word,as.numeric(d[1:10,]$doc1)))
colnames(dftopic)<- c("Word","Frequency")

# for ggplot to understand the order of words, you need to specify factor order

dftopic$Word <- factor(dftopic$Word, levels = dftopic$Word[order(dftopic$Frequency)])
ggplot(dftopic, aes(x=Word, y=Frequency)) + 
  geom_bar(stat="identity", fill="lightgreen", color="grey50")+
  coord_flip()+
  ggtitle("Topic 1")

#Topic 2
newdata2 <-wordMatrix[order(-wordMatrix$doc2),] 
head(newdata2)

d2 <- newdata2
dftopic2 <- as.data.frame(cbind(d[1:15,]$word,as.numeric(d[1:15,]$doc2)))
colnames(dftopic2)<- c("Word","Frequency")
dftopic2$Word <- factor(dftopic2$Word, levels = dftopic2$Word[order(dftopic2$Frequency)])
ggplot(dftopic2, aes(x=Word, y=Frequency)) + 
  geom_bar(stat="identity", fill="lightgreen", color="grey50")+
  coord_flip()+
  ggtitle("Topic 2")

#Topic 3
newdata3 <-wordMatrix[order(-wordMatrix$doc3),] 
head(newdata3)

d3 <- newdata3
dftopic3 <- as.data.frame(cbind(d[1:15,]$word,as.numeric(d[1:15,]$doc3)))
colnames(dftopic3)<- c("Word","Frequency")
dftopic3$Word <- factor(dftopic3$Word, levels = dftopic3$Word[order(dftopic3$Frequency)])
ggplot(dftopic3, aes(x=Word, y=Frequency)) + 
  geom_bar(stat="identity", fill="lightgreen", color="grey50")+
  coord_flip()+
  ggtitle("Topic 3")