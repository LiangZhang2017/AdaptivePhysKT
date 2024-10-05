
import numpy as np
import sklearn.metrics as metrics
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymer4.models import Lmer, Lm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

'''
Prepare R environment using rpy2
'''
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri, numpy2ri

class PFA(object):
    def __init__(self,config):
        self.metrics=config['metrics']
        self.flags=config['flags']
        self.KCmodel=config['KCmodel']
        self.KCs=config['KCs']
        self.KC_used=config['KC_used']
        self.train_data=config['train_data']
        self.test_data=config['test_data']
        self.CV=config['CV']
        self.exper_data=config['exper_data']

        self.is_cross_validation=config['is_cross_validation']

        self.ComputeFeatures=config['ComputeFeatures']

    def training(self):
        print("training")

        utils = importr("utils")
        utils.chooseCRANmirror(ind=1)
        # packnames = ('lme4', 'lmerTest', 'emmeans', 'base', 'pandas2ri')
        # utils.install_packages(StrVector(packnames))
        # utils.install_packages('lme4')  #You have to install the lme4 package at the firs running
        # utils.install_packages('base')
        # utils.install_packages('pandas2ri')
        # utils.install_packages('stats')
        # utils.install_packages('cvms')
        # utils.install_packages('LKT')
        # utils.install_packages('e1071')
        # utils.install_packages('glm2')

        '''
        Reference:
        Cross-validation with groupdata2 for lmer, https://cran.r-project.org/web/packages/groupdata2/vignettes/cross-validation_with_groupdata2.html
        Pymer4: http://eshinjolly.com/pymer4/index.html
        Generalized Linear Mixed Effects Models
        MIXED EFFECT REGRESSION: https://www.pythonfordatascience.org/mixed-effects-regression-python/
        '''

        lme4 = importr('lme4')
        base = importr('base')
        pandas2ri.activate()
        numpy2ri.activate()
        stats=importr('stats')
        cvms = importr('cvms')
        LKT=importr('LKT')
        e1071=importr('e1071')
        glm2=importr('glm2')

        if self.ComputeFeatures is True:
            if self.is_cross_validation is False:
                '''
                Use the lme4.glmer model
                '''
                if self.flags == 'Simple':
                    if self.KCs == 1:
                        model = lme4.glmer("CF_ansbin ~ CF_cor + CF_incor + (1|Student_Id)", data=self.train_data, family="binomial")
                    else:
                        model = lme4.glmer("CF_ansbin ~ CF_cor + CF_incor + (1|" + self.KC_used + ")+ (1|Student_Id)", data=self.train_data,
                                           family="binomial")
                if self.flags == "Full":
                    if self.KCs == 1:
                        # model = lme4.glmer("CF_ansbin ~ CF_cor + CF_incor + (1|Student_Id)", data=self.train_data, family="binomial")
                        mode= Lmer("CF_ansbin ~ CF_cor + CF_incor + (1|Student_Id)", data=self.train_data, family="binomial")
                    else:
                        # model = lme4.glmer("CF_ansbin ~ CF_cor:" + self.KC_used + "+ CF_incor:" + self.KC_used + "+ (1|Student_Id)",
                        #                    data=self.train_data, family="binomial")
                        model = Lmer("CF_ansbin ~ CF_cor:" + self.KC_used + "+ CF_incor:" + self.KC_used + "+ (1|Student_Id)",
                            data=self.train_data, family="binomial")

                if self.is_cross_validation is False:
                    # print(robjects.r.logLik(model))

                    # pred = np.array((robjects.r.predict(model, data=self.train_data, type="response")))
                    # obs = np.array(self.train_data['CF_ansbin'])
                    # self.metrics = self.evaluate(pred, obs)

                    print("model")

        else:
            if self.is_cross_validation is True:
                if self.flags == 'Simple':
                    if self.KCs == 1:
                        # model = robjects.r.glm("CF_ansbin ~ CF_cor + CF_incor + (1|Student_Id)", data=self.train_data,
                        #                    family="binomial")
                        model = Lmer("CF_ansbin ~ CF_cor + CF_incor + (1|Student_Id)", data=self.train_data,
                                               family="binomial")
                    else:
                        # model = robjects.r.glm("CF_ansbin ~ CF_cor + CF_incor + (1|" + self.KC_used + ")+ (1|Student_Id)",
                        #                    data=self.train_data,
                        #                    family="binomial")
                        model = Lmer("CF_ansbin ~ CF_cor + CF_incor + (1|" + self.KC_used + ")+ (1|Student_Id)",data=self.train_data,
                                           family="binomial")

                if self.flags == "Full":
                    if self.KCs == 1:
                        model = lme4.glmer("CF_ansbin ~ CF_cor + CF_incor + (1|Student_Id)", data=self.train_data,
                                           family="binomial")

                        # model = Lmer("CF_ansbin ~ CF_cor + CF_incor + (1|Student_Id)", data=self.train_data,
                        #                        family="binomial")
                    else:
                        # model = robjects.r.glm(
                        #     "CF_ansbin ~ CF_cor:" + self.KC_used + "+ CF_incor:" + self.KC_used + "+ (   1|Student_Id)",
                        #     data=self.train_data, family="binomial")

                        formula="CF_ansbin ~ CF_cor:" + self.KC_used + "+ CF_incor:" + self.KC_used + "+ (1|Student_Id)"
                        # formula="CF_ansbin ~ CF_cor + CF_incor + (1|" + self.KC_used + ")+ (1|Student_Id)"

                        # model=smf.glm(formula=formula,data=self.train_data,family=sm.families.Binomial())
                        # model = Lmer(formula=formula, data=self.train_data, family="binomial")

                        model = lme4.glmer(formula=formula, data=self.train_data, family="binomial")

                # result=model.fit()
                # pred=model.fits

                rstring = ("""function(model,new){
                    out <- predict(model,new,allow.new.levels=TRUE,type='response'"""+""")
                    out
                    }""")

                theTrainData=self.train_data.copy()
                theTestData=self.test_data.copy()
                theKC_used=str(self.KC_used)

                # print(np.unique(theTrainData[theKC_used]))
                # print(np.unique(theTestData[theKC_used]))
                #
                # print("rstring is {}".format(rstring))

                pred_fun=robjects.r(rstring)
                preds=np.array(pred_fun(model,self.test_data))

                #print("preds is {}".format(preds))

                print(base.summary(model))
                # print(model.fits)
                # pred=model.predict(data=self.train_data,skip_data_checks=True)

                obs = np.array(self.test_data['CF_ansbin'])
                # print("obs is {}".format(obs))
                self.metrics=self.evaluate(preds,obs)

    def evaluate(self,pred_data, obs_data):
        mae = metrics.mean_absolute_error(obs_data, pred_data)
        mse = metrics.mean_squared_error(obs_data, pred_data)
        rmse = np.sqrt(mse)
        auc = metrics.roc_auc_score(obs_data, pred_data)

        return mae, rmse, auc

    def Cost(self):
        print("Cost")