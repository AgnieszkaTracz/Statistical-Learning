sleep <- read.csv("Sleep_Efficiency.csv")
sleep <- sleep[,-1] # usuniecie id
sleep <- sleep[,-4] #usuniecie godziny wstania
sleep <- na.omit(sleep)
sleep$Sleep.efficiency <- sleep$Sleep.efficiency*100 # przeskalowanie wartości z ulamków na liczby całkowite
sleep$Gender <- factor(sleep$Gender)
sleep$Smoking.status <- factor(sleep$Smoking.status)
#deklarowanie godziny pójścia spać jako zmiennej jakościowej uporządkowanej
sleep$Bedtime <- as.POSIXct(sleep$Bedtime, format = "%Y-%m-%d %H:%M:%S")
sleep$Bedtime <- format(sleep$Bedtime, "%H:%M")
levels(factor(sleep$Bedtime))
levels <- c("21:00", "21:30", "22:00", "22:30", "23:00", "00:00", "00:30", "01:00", "01:30", "02:00", "02:30")
sleep$Bedtime <- factor(sleep$Bedtime, levels = levels, ordered = T)
sleep$Bedtime
#uporządkowana zmienna jakościowa: przebudzenia
levels <- levels(factor(sleep$Awakenings))
sleep$Awakenings <- factor(sleep$Awakenings, levels = levels, ordered = T)
sleep$Awakenings
#uporządkowana zmienna jakościowa: częstość ćwiczeń w tygodniu
levels <- levels(factor(sleep$Exercise.frequency))
sleep$Exercise.frequency <- factor(sleep$Exercise.frequency, levels = levels, ordered = T)
sleep$Exercise.frequency

#############################################################
## wstępna analiza

#podstawowe statystyki naszego zbioru danych
summary(sleep)
str(sleep)

attach(sleep)
boxplot(Sleep.efficiency ~ Awakenings) # im więcej przebudzeń tym gorsza jakość snu
boxplot(Sleep.efficiency ~ Sleep.duration) # brak związku długości snu z efektywnością
boxplot(Sleep.efficiency ~ Gender) # brak związku płci z efektywnością
boxplot(Sleep.efficiency ~ Caffeine.consumption) # brak związku spożycia kofeiny z efektywnością snu lub nawet pozytywny wpływ kofeiny na sen
boxplot(Sleep.efficiency ~ Smoking.status) # nieznacznie negatywny wpływ palenia na sen
boxplot(Sleep.efficiency ~ Alcohol.consumption) # negatywny wpływ spożycia alkoholu na sen
boxplot(Sleep.efficiency ~ Exercise.frequency) #pozytywny wpływ częstej aktywności fizycznej

library(ggplot2)
library(gridExtra)

plot1 <- ggplot(sleep, aes(x = Age, y = Sleep.efficiency)) +
  geom_bar(stat = "summary", fun = "mean")

plot2 <- ggplot(sleep, aes(x = REM.sleep.percentage, y = Sleep.efficiency)) +
  geom_bar(stat = "summary", fun = "mean") 

plot3 <- ggplot(sleep, aes(x = Deep.sleep.percentage, y = Sleep.efficiency)) +
  geom_bar(stat = "summary", fun = "mean") 

plot4 <- ggplot(sleep, aes(x = Light.sleep.percentage, y = Sleep.efficiency)) +
  geom_bar(stat = "summary", fun = "mean") 

plot5 <- ggplot(sleep, aes(x = Bedtime, y = Sleep.efficiency)) +
  geom_bar(stat = "summary", fun = "mean") 

grid.arrange(plot1, plot2, plot3, plot4, plot5, ncol = 2) #brak wpływu wieku, długości fazy REM i pory pójścia spać na sen

grid.arrange( plot3, plot4, ncol = 2) # wyraźnie widoczny związek długiej (ok. 60% całości snu) fazy snu głębokiego i krótkiej (20%) fazy snu płytkiego z wysoką efektywnością snu

detach(sleep)

#########################################################
## analiza właściwa

# numery indeksów według których będziemy dzielić dane na treningowe i testowe
set.seed(1)
train <- sample(1:nrow(sleep), 300)
test <- (-train)

x <- model.matrix(Sleep.efficiency ~ ., sleep)[,-1] #macierz predyktorów
y <- sleep$Sleep.efficiency #kolumna zmiennej zależnej

y.test <- y[test]


###################################################
##funkcja błędu RMSE przy pomocy którego będziemy porównywali modele

RMSE <- function(x, y) {
  sqrt(mean((x-y)^2))
}
###################################################
## wybór najlepszego podzbioru:

library(leaps)

# Selekcja krokowa
# ----------------------------------------
# regsubsets() - wybór najlepszego podzbioru określonej liczby predyktorów

# funkcja predict() z zajęć
predict.regsubsets <- function(object, newdata, id, ...){
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[ , xvars] %*% coefi
}
# ----------------------------------------

# optymalny model z użyciem 10-CV
k <- 10
n <- nrow(sleep)
ncomps <- 27
set.seed(1)
folds <- sample(rep(1:k, length = n))

# macierz, w której będziemy przechowywali błędy RMSE:
cv.errors <- matrix(NA, k, ncomps, dimnames = list(NULL, paste(1:ncomps)))

# cross-validation:
for(j in 1:k){
  best.fit <- regsubsets(Sleep.efficiency ~ ., data = sleep[folds != j, ], nvmax = ncomps)
  for(i in 1:ncomps){
    pred <- predict(best.fit, sleep[folds == j, ], id = i)
    cv.errors[j, i] <- sqrt(mean((sleep$Sleep.efficiency[folds == j] - pred)^2))
  }
}

# W macierzy cv.errors element (j, i) odpowiada błędowi RMSE
# dla j-tego podzbioru z procedury CV dla najlepszego modelu z i zmiennymi
# Liczymy średnie z kolumn:
mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors

par(mfrow = c(1, 1))
plot(mean.cv.errors, type = "b")
reg.best.ncomps <- which.min(mean.cv.errors) # liczba zmiennych dla ktorej dostajemy najmniejszy bład
abline(v=reg.best.ncomps)
# 10-CV wybrała model z 16 zmiennymi

#bład na teście
reg.best <- regsubsets(Sleep.efficiency ~ ., data = sleep, subset = train,  nvmax = reg.best.ncomps)
reg.pred <- predict(reg.best, sleep[test, ], id = reg.best.ncomps)
RMSE(reg.pred, y.test) # 6.72

# Budujemy najlepszy model z 16 zmiennymi na pełnym zbiorze danych:
reg.best <- regsubsets(Sleep.efficiency ~ ., data = sleep, nvmax = reg.best.ncomps)
coef(reg.best, reg.best.ncomps)
reg.pred <- predict(reg.best, sleep[test, ], id = reg.best.ncomps)
RMSE(reg.pred, y.test)

######################################################################
# Regresje (przy użyciu glmnet)

#######################################################
## regresja grzbietowa (alpha = 0, lambda>0)
library(glmnet)
ridge.mod <- glmnet(x[train, ], y[train], alpha = 0)

### estymacja przez średnią
RMSE(mean(y[train]), y.test) #13.63

# regresja liniowa MNK (lambda = 0):
summary(lm(y ~ x, subset = train)) # które współczynniki są istotne
ridge.pred <- predict(ridge.mod, s = 0, newx = x[test , ],
                      exact = T, x = x[train, ], y = y[train ])
RMSE(ridge.pred, y.test) #6.13

predict(ridge.mod, s = 0, exact = T, type = "coefficients",
        x = x[train, ], y = y[train]) #współczynniki dla naszego modelu liniowego

## wybór lambdy walidacją krzyżową
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam #1.20

# regresja grzbietowa i RMSE dla najlepszej lambdy:
ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test, ])
RMSE(ridge.pred, y.test) # 6.11

predict(ridge.mod, type = "coefficients", s = bestlam) # Wszystkie współczynniki są niezerowe



#------------------------------------------------------------------
# Lasso
# -----------------------------------------------------------------

# alpha = 1 dla lassso
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1)
plot(lasso.mod, "lambda")   # Liczba współczynników w modelu w zależności od lambda

# Szukamy lambdy za pomocą CV:
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min

# RMSE na zbiorze testowym:
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])
RMSE(lasso.pred, y.test) #6.12

# Współczynniki optymalnego modelu
out <- glmnet(x, y, alpha = 1)
predict(out, type = "coefficients", s = bestlam)


#########################################################################
# PCR

library(pls)
# model na zbiorze testowym
set.seed(1)
pcr.fit <- pcr(Sleep.efficiency ~., data = sleep, subset = train, scale = TRUE, validation = "CV") # scale = TRUE - standaryzacja cech przed PCA

validationplot(pcr.fit, val.type = "RMSEP")
abline(v=25, col = "red")
(pcr.best.ncomps <- which.min(RMSEP(pcr.fit)$val[1,,]) -1 ) # najmniejszy błąd dla 25 predyktorów (25=26 - 1 wyraz wolny)
RMSEP(pcr.fit)$val[1,,]

# Liczymy RMSE na zbiorze testowym
pcr.pred <- predict(pcr.fit, x[test, ], ncomp = pcr.best.ncomps)
RMSE(pcr.pred, y.test) # 6.08

#model na całym zbiorze
pcr.fit <- pcr(y ~ x, scale = TRUE, ncomp = pcr.best.ncomps)
summary(pcr.fit) #przy 25 zmiennych wyjaśnione 84% wariancji

# PLS
# ----------------------------------------

set.seed(1)
pls.fit <- plsr(Sleep.efficiency ~ ., data = sleep, subset = train,
                scale = TRUE , validation = "CV")
summary(pls.fit)

validationplot(pls.fit, val.type = "RMSEP")

(pls.best.ncomps <- which.min(RMSEP(pls.fit)$val[1,,]) -1 ) # najmniejszy błąd dla 3 predyktorów (3 = 4 - 1 wyraz wolny)
RMSEP(pls.fit)$val[1,,]

# błąd RMSE na zbiorze testowym
pls.pred <- predict(pls.fit, x[test, ], ncomp = pls.best.ncomps)
RMSE(pls.pred, y.test) # 6.01

# Na koniec budujemy model PLS z M = 3 na całym zbiorze danych:
pls.fit <- plsr(Sleep.efficiency ~ ., data = sleep, scale = TRUE, ncomp = pls.best.ncomps)
summary(pls.fit) # przy trzech składowych wyjaśnione 84% wariancji Sleep.efficiency (tyle samo co w pcr)

################################################################
## metody drzewiaste

library(tree)
set.seed(1)

#pojedyncze drzewo
tree.sleep <- tree(Sleep.efficiency ~ ., sleep, subset = train)
summary(tree.sleep)
plot(tree.sleep)
text(tree.sleep, pretty = 0)

pred.tree.sleep <- predict(tree.sleep, newdata = sleep[test, ]) #predykcje dla zbioru testowego
eff.test.sleep <- sleep[test, 5] # wartości efektywności snu na zbiorze testowym
plot(pred.tree.sleep, eff.test.sleep)
abline(0, 1)
RMSE(pred.tree.sleep, eff.test.sleep) #RMSE = 5.26

# sprawdzenie czy opłaca się przycinać drzwo
cv.sleep <- cv.tree(tree.sleep)
plot(cv.sleep$size, cv.sleep$dev, type = "b") #błąd najniższy przy pełnym drzeiwe, ale sprawdzamy jeszcze jaki błąd dostaniemy po obciąciu do trzech liści

pruned.sleep <- prune.tree(tree.sleep, best = 3)
plot(pruned.sleep) #obcięte drzewo
text(pruned.sleep, pretty = 0)

#RMSE na zbiorze testowym dla obcietego drzewa
pred.pruned.sleep <- predict(pruned.sleep, newdata = sleep[-train, ])
plot(pred.pruned.sleep, eff.test.sleep)
abline(0, 1)
RMSE(pred.pruned.sleep, eff.test.sleep) #5.42

##################################
## bagging & random forest

library(randomForest)
set.seed(1)
# najpierw budujemy domyślny model baggingowy (mtry=12, ntree=500)
bag.sleep <- randomForest(Sleep.efficiency ~ ., data = sleep,
                          subset = train,
                          mtry = 12, importance = TRUE)
bag.sleep
#bład na zbiorze testowym
pred.bag.sleep <- predict(bag.sleep, newdata = sleep[-train, ])
plot(pred.bag.sleep, eff.test.sleep)
abline(0, 1) 
RMSE(pred.bag.sleep, eff.test.sleep) #4.83

rf.rmse <- matrix(NA, 12, 12, dimnames = list(1:12, (1:12)*25)) # macierz, którą będziemy uzupełniali wartościami błedów RMSE; w wierszach będą wartosci dla parametrów mtry a w kolumnach wartosc dla ntree

for (k in 1:12) { # bagging z różną liczbą drzew
  bag.sleep <- randomForest(Sleep.efficiency ~ ., data = sleep,
                            subset = train, mtry = 12, ntree = k*25)
  pred.bag.sleep <- predict(bag.sleep, newdata = sleep[-train, ])
  rf.rmse[12,k] <- RMSE(pred.bag.sleep, eff.test.sleep)
  cat(paste0("ntree = ", k*25, "\t RMSE = ", rf.rmse[12,k], "\n"))
}
# liczba drzew dajaca najmniejszy RMSE
matrix(c(order(rf.rmse[12,])*25, rf.rmse[12,order(rf.rmse[12,])]),
       ncol = 2, dimnames = list(NULL, c("ntree", "RMSE"))) #najmniejszy blad = 4,80 dla 225 drzew

for (i in 1:11) { #rf z roznymi wartosciami mtry
  for (j in 1:12) { # oraz z rozna liczba drzew
    rf.sleep <- randomForest(Sleep.efficiency ~ ., data = sleep,
                              subset = train, mtry = i, ntree = j*25)
    pred.rf.sleep <- predict(rf.sleep, newdata = sleep[-train, ])
    rf.rmse[i,j] <- RMSE(pred.rf.sleep, eff.test.sleep)
  }
}

rf.rmse

#macierz z uporządkowanymi od najmniejszej wartości błędami i parametrami, przy których dane błędy sa osiągane
rf.min.rmse <- t(sapply(1:12, function(i) which(rf.rmse == sort(rf.rmse)[i], arr.ind = TRUE)))
rf.min.rmse <- matrix(c(rf.min.rmse[,1], 25*rf.min.rmse[,2],
                      sort(rf.rmse)[1:12]), ncol = 3,
                      dimnames = list(NULL, c("mtry", "ntree", "RMSE") ) )
# parametry mtry, ntree dla 12 najmniejszych wartosci RMSE
rf.min.rmse # min RMSE = 4.59

#########################################################
## Boosting

library(gbm)
set.seed(1)
## przykladowy model
boost.sleep <- gbm(Sleep.efficiency ~ ., data = sleep[train, ],
                   distribution = "gaussian", n.trees = 5000,
                   interaction.depth = 2, shrinkage = 0.5)

summary(boost.sleep)

# partial dependence plots #wynik w zależności od wybranych zmiennych
plot(boost.sleep, i = "Caffeine.consumption")
plot(boost.sleep, i = "Awakenings")
plot(boost.sleep, i = "Deep.sleep.percentage")

#RMSE dla powyższego przykładowego modelu
pred.boost.sleep <- predict(boost.sleep, newdata = sleep[test, ], n.trees = 500)
RMSE(pred.boost.sleep, eff.test.sleep) #5.77

# porownanie 36 modeli z roznymi parametrami shrinkage oraz interaction.depth
boost.rmse <- c()
for (shrink in c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5)) { #siatka wartości dla parametru schrinkage
  for (dep in 1:6) { #poziomy dla których będziemy rozpatrywać parametr interaction.depth
    boost.sleep <- gbm(Sleep.efficiency ~ .,
                       data = sleep[train, ],
                       distribution = "gaussian",
                       n.trees = 5000,
                       interaction.depth = dep,
                       shrinkage = shrink,
                       cv.folds = 5,
                       train.fraction = 0.8)
    boost.ntrees = gbm.perf(boost.sleep, method = "cv") #walidacja metodą krzyżową
    pred.boost.sleep <- predict(boost.sleep, #waznaczanie modelu ze względu na optymalną wartość parametru n.trees
                                newdata = sleep[-train, ],
                                n.trees = boost.ntrees)
    boost.rmse <- c(boost.rmse, shrink, dep, boost.ntrees, RMSE(pred.boost.sleep, eff.test.sleep)) #zapisywanie RMSE dla konkretnych parametrów
    cat(paste0("shrink = ", shrink,
               "\t depth = ", dep,
               "\t n.trees = ", boost.ntrees,
               "\t RMSE = ", tail(boost.rmse, n=1), "\n"))
  }
}
boost.rmse <- matrix(boost.rmse, byrow = TRUE, ncol = 4, dimnames = list(NULL, c("shrinkage", "depth", "n.trees", "RMSE"))) #przekształcenie wektora z zapisanymi błędami do postaci macierzy
boost.rmse

#tablica w wymiarami RMSE dla konkretnych parametrów,
#parametry dla 12 najmniejszych wartosci RMSE
boost.min.rmse <- boost.rmse[ order(boost.rmse[, 4])[1:12] , ]
boost.min.rmse # min RMSE = 4.94

#####################################################
## BART

library(BART)

# deklarowanie zmiennych nazywających zbiór treningowy i testowy (podział na te zbiory taki sam od początku, zadany przez indeksy dla traina)
x <- sleep[, -5]
y <- sleep[, "Sleep.efficiency"]
xtrain <- x[train, ]
ytrain <- y[train]
xtest <- x[test, ]
ytest <- y[test]

set.seed(1)
#budowa BART
bart.sleep <- gbart(xtrain, ytrain, x.test = xtest)
#bład prognozy
pred.bart.sleep <- bart.sleep$yhat.test.mean
RMSE(pred.bart.sleep, eff.test.sleep) # 5.22

# Można sprawdzić, ile razy, średnio po wszystkich iteracjach,
# każda ze zmiennych pojawiła się w całym zbiorze drzew:
ord <- order(bart.sleep$varcount.mean, decreasing = T)
bart.sleep$varcount.mean[ord]

######################################################################
#XGBoost

library(xgboost)
# deklarowanie zmiennych nazywających zbiór treningowy i testowy (podział na te zbiory taki sam od początku, zadany przez indeksy dla traina)
samples <- train

train <- sleep[samples,]
test <- sleep[-samples,]

train_x <- data.matrix(train[, -5])   # pakiet xgboost używa matrix data
train_y <- train[,5]

test_x <- data.matrix(test[, -5])
test_y <- test[, 5]

# przekształcenie do formatu optymalnego dla modeli XGBoost
xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)

# watchlist - lista zawierająca zestawy danych, na których chcemy śledzić
# błędy podczas kolejnych iteracji uczenia modelu
watchlist <- list(train=xgb_train, test=xgb_test)
watchlist

#budowa modelu
model <- xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist, nrounds = 200)
# nrounds - maksymalna liczba iteracji, tutaj 200, bo zbiór mały,

#szukamy liczby iteracji dla której RMSE jest najmniejszy
(best.nrounds <- which.min(model$evaluation_log$test_rmse))
model$evaluation_log$test_rmse[best.nrounds]

# Na ostateczny model wybierzemy zatem ten po 29 iteracjach:
final <- xgb.train(data = xgb_train, max.depth = 3, nrounds = best.nrounds, verbose = 0)
# błąd na zbiorze testowym
pred_y <- predict(final, xgb_test)
RMSE(pred_y,test_y) #4.66

###########################################################################
## porownanie bledow

## selekcja krokowa = 6.72
## estymacja srednią = 13.25
## regresja liniowa = 6.13
## regresja grzbietowa = 6.11
## regresja lasso = 6.12
## pcr = 6.08
## pls = 6.01
## tree = 5.26
## bagging min RMSE = 4.8
## rf min RMSE = 4.59
## boosting min RMSE = 4.94
## bart = 5.22
## xgb = 4.66
