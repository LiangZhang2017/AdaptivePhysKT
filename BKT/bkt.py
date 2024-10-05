

import pandas as pd
import numpy as np
import math
import csv
from sklearn.metrics import mean_squared_error,mean_absolute_error,roc_auc_score

class ClassicBKT():
    def __init__(self):
        self.paramChoices = []
        self.best_params = {}

        self.activities = {}  # {activity: [skill1, skill2, skill3, ...]}
        self.activity_d = {}  # (activity: index)
        self.n_activities = 0

        self.skill_d = {}  # (skill: index)
        self.skills = []  # [skill1, skill2, skill3, ...]
        self.n_skills = 0
        self.output_path=''

        print("BKT initialized")

    # Generate Parameters brute force: run it once for the entire model
    def generateParams(self,output_path):
        print("gran")
        gran=20
        spread=[x/float(gran) for x in range(1,gran)]

        for p_guess in [x for x in spread if x<0.5]:
            for p_slip in [x for x in spread if x<0.5]:
                for p_init in spread:
                    for p_transit in spread:
                        self.paramChoices.append(
                            {'p_guess':p_guess,
                             'p_slip':p_slip,
                             'p_init':p_init,
                             'p_transit':p_transit
                            }
                        )

        self.output_path=output_path

    def generateSkillMap(self,exper_data):
        # print(pd.concat([exper_data['Question_Id'],exper_data['KC_Theoretical_Levels']]).unique())
        #column_values = exper_data[["Question_Id", "KC_Theoretical_Levels"]].values

        skill_map=exper_data.groupby(['Question_Id', 'KC_Theoretical_Levels']).size().reset_index().rename(columns={0: 'count'})

        for index, activity in enumerate(skill_map['Question_Id']):
            self.activity_d[activity]=index
            skill=skill_map['KC_Theoretical_Levels'][index]
            self.skill_d[skill] = index
            self.skills.append(skill)
            self.activities[activity]=[skill]
            self.n_activities += 1
            self.n_skills += 1

    def fit(self, trainingSet, index):
        scores = [[] for _ in range(self.n_skills)]
        user_scores = [[] for _ in range(self.n_skills)]
        unique_users = set()
        final_mastery = {}  # Store final mastery probabilities for each user-skill combination during training

        print("Training set is", trainingSet)

        # Step 1: Organize data by users and activities
        for row in trainingSet:
            user = row[0]
            score = row[3]
            activity = row[1]

            if user not in unique_users:
                unique_users.add(user)

                # Copy the scores for each skill
                for s in range(self.n_skills):
                    if len(user_scores[s]) > 0:
                        scores[s].append(list(user_scores[s]))
                user_scores = [[] for _ in range(self.n_skills)]

            # Store scores for skills in the current activity
            for skill in self.activities[activity]:
                user_scores[self.skill_d[skill]].append(score)

        # Copy the scores to the main scores array
        for s in range(self.n_skills):
            if len(user_scores[s]) > 0:
                scores[s].append(list(user_scores[s]))

        # Initialize best_params, best_scores, and n_questions for each skill
        best_params = [self.paramChoices[0] for _ in range(self.n_skills)]
        best_scores = [float("-inf") for _ in range(self.n_skills)]
        n_questions = [0 for _ in range(self.n_skills)]

        print("Initial best_params:", best_params)
        print("Initial best_scores:", best_scores)
        print("Initial n_questions:", n_questions)

        # Step 2: Fit the model for each skill and find final mastery for each user
        for c in range(self.n_skills):
            best_params[c] = self.paramChoices[0]

            # Iterate through parameter choices
            for p, param in enumerate(self.paramChoices):
                total_score = 0
                n_attempts = 0

                # Iterate through users and calculate user-specific mastery and scores
                for u in range(len(scores[c])):
                    user = list(unique_users)[u]
                    n_attempts += len(scores[c][u])
                    p_guess = param['p_guess']
                    p_guess_c = 1 - p_guess
                    p_slip = param['p_slip']
                    p_slip_c = 1 - p_slip
                    p_transit = param['p_transit']

                    # Initialize mastery state
                    user_score = 0
                    p_mastered = param['p_init']
                    p_mastered_c = 1 - p_mastered

                    # Iterate over each user's scores
                    for s in scores[c][u]:
                        sm1 = s - 1
                        p_correct = (p_mastered * p_slip_c) + (p_mastered_c * p_guess)

                        # Clamp probability values to avoid log errors
                        p_correct = max(0.0001, min(0.9999, p_correct))

                        # Compute log likelihood
                        user_score += math.log((s * p_correct) - (sm1 * (1 - p_correct)))

                        # Update learning probability
                        p_learn = (
                                      (s * p_mastered * p_slip_c / (p_mastered * p_slip_c + p_mastered_c * p_guess))
                                  ) - (
                                      (sm1 * p_mastered * p_slip / (p_mastered * p_slip + p_mastered_c * p_guess_c))
                                  )
                        p_mastered = p_learn + (1 - p_learn) * p_transit
                        p_mastered_c = 1 - p_mastered

                    total_score += user_score

                    # Save final mastery probability for the user and skill
                    if user not in final_mastery:
                        final_mastery[user] = {}
                    final_mastery[user][self.skills[c]] = p_mastered

                # Update best_params and best_scores if the current set of parameters is better
                if total_score > best_scores[c]:
                    best_scores[c] = total_score
                    best_params[c] = self.paramChoices[p]
                    n_questions[c] = n_attempts

        # Save the best overall parameters per skill (this is still necessary for general performance tracking)
        for c, skill in enumerate(self.skills):
            print(f"Best score for skill {skill}: {best_scores[c]}/{n_questions[c]}")
            self.best_params[skill] = best_params[c]

        # Save final mastery and corresponding BKT parameters for each user-skill pair during training
        train_test_mode = "train"
        self.save_params_to_csv(train_test_mode, index)
        self.save_user_params(train_test_mode, final_mastery, index)

    def save_params_to_csv(self, train_test_mode, index):
        parameter_writer = csv.writer(
            open(self.output_path + '/' + 'skills_parameters_' + train_test_mode + '_' + str(index) + '.csv', 'w'))
        parameter_writer.writerow(['skill', 'p_init', 'p_transit', 'p_guess', 'p_slip', 'p_mastered'])

        # For each skill, calculate and store the final mastered probability
        for skill in self.skills:
            params = self.best_params[skill]

            # Compute the final mastered probability based on the learned transition dynamics
            p_init = params['p_init']
            p_transit = params['p_transit']
            p_mastered = p_init + (1 - p_init) * p_transit  # Final mastery calculation logic

            row = [skill, str(params['p_init']), str(params['p_transit']),
                   str(params['p_guess']), str(params['p_slip']), str(p_mastered)]
            parameter_writer.writerow(row)

    def save_user_params(self, train_test_mode, final_mastery, index):
        """ Save final mastered probabilities for each user along with BKT parameters and p_mastered into a CSV file. """
        parameter_writer = csv.writer(
            open(self.output_path + '/' + 'user_skills_final_mastery_' + train_test_mode + '_' + str(index) + '.csv',
                 'w'))
        # Add column headers
        parameter_writer.writerow(
            ['user', 'skill', 'p_init', 'p_transit', 'p_guess', 'p_slip', 'p_mastered', 'final_mastered'])

        # Loop through the final mastery dictionary and save the corresponding BKT parameters
        for user, skills_mastery in final_mastery.items():
            for skill, final_mastered in skills_mastery.items():
                # Get the BKT parameters for the skill from the `best_params` dictionary
                if skill in self.best_params:
                    params = self.best_params[skill]
                    p_init = params['p_init']
                    p_transit = params['p_transit']
                    p_guess = params['p_guess']
                    p_slip = params['p_slip']

                    # Calculate p_mastered similar to how final_mastered is calculated in save_params_to_csv
                    # Using the transition dynamic formula
                    p_mastered = p_init + (1 - p_init) * p_transit
                else:
                    # Handle the case where no parameters are found (shouldn't happen normally)
                    p_init = p_transit = p_guess = p_slip = p_mastered = 'NaN'

                # Write user, skill, parameters, p_mastered, and final_mastered to the CSV
                row = [user, skill, str(p_init), str(p_transit), str(p_guess), str(p_slip), str(p_mastered),
                       str(final_mastered)]
                parameter_writer.writerow(row)

    def writePrediction(self,testSet,index):
        print("write Prediction")

        y,scores,y_true,y_scores,final_mastery=self.predict(testSet)

        writer = csv.writer(open(self.output_path+'/'+'results' + str(index) + '.csv', 'w'))
        writer.writerow(['y', 'prediction'])
        for user in y.keys():  # number of users
            for i in range(0, len(y[user])):
                line = [str(y[user][i]), str(scores[user][i])]
                writer.writerow(line)
                # if y[user][i] == 0:
                #     # print('y: ' + str(y[user][i]))
                #     # print('score: ' + str(scores[user][i]))
            writer.writerow([])

        # auc=0
        rmse=0
        mae=0

        # user_auc=roc_auc_score(y_true,y_scores)
        user_rmse = mean_squared_error(y_true, y_scores, squared=False)
        user_mae = mean_absolute_error(y_true, y_scores)
        # auc += user_auc
        rmse += user_rmse
        mae += user_mae

        writer = csv.writer(open(self.output_path + '/' + 'Metrics_Results' + str(index) + '.csv', 'w'))
        writer.writerow(['Index', 'Metric','Value'])
        # line1=[str(index),'AUC',auc]
        line2 =[str(index), 'rmse', rmse]
        line3=[str(index), 'mae', mae]
        # writer.writerow(line1)
        writer.writerow(line2)
        writer.writerow(line3)

        # Save skill-specific and user-specific parameters for the test dataset
        self.save_params_to_csv("test", index)
        self.save_user_params("test", final_mastery, index)

        return mae,rmse

    def predict(self, testSet):
        test_scores=[{} for c in range(0,self.n_skills)]
        user_scores=[[] for c in range(0,self.n_skills)]

        p_correct_d={}
        unique=[]

        final_mastery = {}  # To store final mastered probabilities for each user

        for r, row in enumerate(testSet):
            user = row[0]
            score = row[3]
            activity = row[1]

            skill_indicies = []

            if not p_correct_d.__contains__(user):
                p_correct_d[user]={}

            if not p_correct_d[user].__contains__(activity):
                p_correct_d[user][activity]=[]

            for skill in self.activities[activity]:
                if not test_scores[self.skill_d[skill]].__contains__(user):
                    test_scores[self.skill_d[skill]][user]=[]
                test_scores[self.skill_d[skill]][user].append(score)
                skill_indicies.append(len(test_scores[self.skill_d[skill]][user])-1)

            p_correct_d[user][activity].append(skill_indicies)

            if r ==len(testSet)-1:
                for s in range(0,self.n_skills):
                    if len(user_scores[s])>0:
                        test_scores[s][user]=user_scores[s]

        p_scores=[{} for c in range(0,self.n_skills)]

        for c,concept in enumerate(self.skills):
            for user in test_scores[c].keys():
                if not self.best_params.__contains__(concept):
                    self.best_params[concept]=self.paramChoices[0]

                p_guess=self.best_params[concept]['p_guess']
                p_guess_c=1-p_guess
                p_slip=self.best_params[concept]['p_slip']
                p_slip_c=1-p_slip
                p_transit=self.best_params[concept]['p_transit']
                p_mastered=self.best_params[concept]['p_init']
                p_mastered_c=1-p_mastered

                p_scores[c][user]=[]
                final_mastered = p_mastered  # To keep track of the final mastery after all interactions

                for s in test_scores[c][user]:
                    sm1=s-1
                    p_correct=(p_mastered*p_slip_c)+(p_mastered_c*p_guess)
                    p_learn=(s*p_mastered*p_slip_c/(p_mastered*p_slip_c+p_mastered_c*p_guess))-(sm1*p_mastered*p_slip/(p_mastered*p_slip+p_mastered_c*p_guess_c))
                    p_mastered=p_learn+(1-p_learn)*p_transit
                    p_mastered_c=1-p_mastered
                    p_scores[c][user].append(p_correct)
                    final_mastered = p_mastered  # Update final mastery probability after the interaction

                # Save the final mastery probability for the user and skill
                if user not in final_mastery:
                    final_mastery[user] = {}
                final_mastery[user][self.skills[c]] = final_mastered

        y={}
        scores={}

        y_true=[]
        y_scores=[]

        for user in p_correct_d:
            y[user]=[]
            scores[user]=[]

            for activity in p_correct_d[user]:
                for attempt in p_correct_d[user][activity]:
                    acc_score=0
                    y_score=0

                    for i in range(0,len(attempt)):
                        index=attempt[i]
                        skill_i=self.skill_d[self.activities[activity][i]]

                        acc_score+=p_scores[skill_i][user][index]
                        if i==0:
                            y_score=test_scores[skill_i][user][index]

                    y[user].append(y_score)
                    y_true.append(y_score)
                    scores[user].append(acc_score/len(self.activities[activity]))
                    y_scores.append(acc_score/len(self.activities[activity]))

        return y,scores,y_true,y_scores, final_mastery