def findDecision(obj): #obj[0]: feature_0, obj[1]: feature_1, obj[2]: feature_2, obj[3]: feature_3, obj[4]: feature_4, obj[5]: feature_5, obj[6]: feature_6, obj[7]: feature_7, obj[8]: feature_8, obj[9]: feature_9
   # {"feature": "feature_8", "instances": 17117, "metric_value": 0.0481, "depth": 1}
   if obj[8]<=27.59173113279196:
      # {"feature": "feature_0", "instances": 10481, "metric_value": 0.0229, "depth": 2}
      if obj[0]<=121.27438066105142:
         # {"feature": "feature_6", "instances": 10085, "metric_value": 0.0089, "depth": 3}
         if obj[6]>-54.56405365689118:
            # {"feature": "feature_4", "instances": 9786, "metric_value": 0.0094, "depth": 4}
            if obj[4]<=0.2000482934804823:
               # {"feature": "feature_1", "instances": 5575, "metric_value": 0.0067, "depth": 5}
               if obj[1]<=33.24735032175879:
                  return 0.9186691312384473
               elif obj[1]>33.24735032175879:
                  return 0.7875354107648725
               else:
                  return 0.9020627802690583
            elif obj[4]>0.2000482934804823:
               # {"feature": "feature_3", "instances": 4211, "metric_value": 0.005, "depth": 5}
               if obj[3]<=0.5080465922583709:
                  return 0.8295780417861532
               elif obj[3]>0.5080465922583709:
                  return 0.7214689265536723
               else:
                  return 0.7841367846117312
            else:
               return 0.8513182096873084
         elif obj[6]<=-54.56405365689118:
            # {"feature": "feature_1", "instances": 299, "metric_value": 0.0575, "depth": 4}
            if obj[1]<=35.78561471571906:
               # {"feature": "feature_9", "instances": 181, "metric_value": 0.0477, "depth": 5}
               if obj[9]>269.9353629834254:
                  return 0.7549019607843137
               elif obj[9]<=269.9353629834254:
                  return 0.3291139240506329
               else:
                  return 0.569060773480663
            elif obj[1]>35.78561471571906:
               # {"feature": "feature_4", "instances": 118, "metric_value": 0.0919, "depth": 5}
               if obj[4]<=0.1354042372881356:
                  return 0.21739130434782608
               elif obj[4]>0.1354042372881356:
                  return 0
               else:
                  return 0.1271186440677966
            else:
               return 0.39464882943143814
         else:
            return 0.8377788795240456
      elif obj[0]>121.27438066105142:
         # {"feature": "feature_6", "instances": 396, "metric_value": 0.0574, "depth": 3}
         if obj[6]<=-33.49648661616162:
            # {"feature": "feature_2", "instances": 225, "metric_value": 0.06, "depth": 4}
            if obj[2]>2.4232:
               # {"feature": "feature_9", "instances": 221, "metric_value": 0.0309, "depth": 5}
               if obj[9]>254.12891809954746:
                  return 0.025
               elif obj[9]<=254.12891809954746:
                  return 0
               else:
                  return 0.013574660633484163
            elif obj[2]<=2.4232:
               return 1
            else:
               return 0.03111111111111111
         elif obj[6]>-33.49648661616162:
            # {"feature": "feature_9", "instances": 171, "metric_value": 0.0547, "depth": 4}
            if obj[9]>145.3557344783373:
               # {"feature": "feature_2", "instances": 144, "metric_value": 0.0391, "depth": 5}
               if obj[2]<=4.112166500523946:
                  return 0.3161764705882353
               elif obj[2]>4.112166500523946:
                  return 1
               else:
                  return 0.3541666666666667
            elif obj[9]<=145.3557344783373:
               return 0
            else:
               return 0.2982456140350877
         else:
            return 0.14646464646464646
      else:
         return 0.8116591928251121
   elif obj[8]>27.59173113279196:
      # {"feature": "feature_0", "instances": 6636, "metric_value": 0.0738, "depth": 2}
      if obj[0]<=53.17349079264617:
         # {"feature": "feature_2", "instances": 4753, "metric_value": 0.0204, "depth": 3}
         if obj[2]<=2.5017190195665897:
            # {"feature": "feature_9", "instances": 2524, "metric_value": 0.0085, "depth": 4}
            if obj[9]<=214.8999880478378:
               # {"feature": "feature_1", "instances": 2093, "metric_value": 0.0039, "depth": 5}
               if obj[1]>7.146848167385758:
                  return 0.7270742358078602
               elif obj[1]<=7.146848167385758:
                  return 0.5555555555555556
               else:
                  return 0.705685618729097
            elif obj[9]>214.8999880478378:
               # {"feature": "feature_5", "instances": 431, "metric_value": 0.0101, "depth": 5}
               if obj[5]<=4.426436658932714:
                  return 0.5688888888888889
               elif obj[5]>4.426436658932714:
                  return 0.36893203883495146
               else:
                  return 0.4733178654292343
            else:
               return 0.6660063391442155
         elif obj[2]>2.5017190195665897:
            # {"feature": "feature_1", "instances": 2229, "metric_value": 0.037, "depth": 4}
            if obj[1]>10.245568323980862:
               # {"feature": "feature_9", "instances": 1902, "metric_value": 0.029, "depth": 5}
               if obj[9]<=153.97233364879074:
                  return 0.6076604554865425
               elif obj[9]>153.97233364879074:
                  return 0.27564102564102566
               else:
                  return 0.4442691903259727
            elif obj[1]<=10.245568323980862:
               # {"feature": "feature_9", "instances": 327, "metric_value": 0.0541, "depth": 5}
               if obj[9]>161.8366470948012:
                  return 0
               elif obj[9]<=161.8366470948012:
                  return 0.06289308176100629
               else:
                  return 0.03058103975535168
            else:
               return 0.3835800807537012
         else:
            return 0.5335577529981065
      elif obj[0]>53.17349079264617:
         # {"feature": "feature_9", "instances": 1883, "metric_value": 0.0187, "depth": 3}
         if obj[9]<=308.6146710901211:
            # {"feature": "feature_1", "instances": 1566, "metric_value": 0.0083, "depth": 4}
            if obj[1]<=44.797436015325665:
               # {"feature": "feature_2", "instances": 981, "metric_value": 0.0109, "depth": 5}
               if obj[2]>2.723866729253481:
                  return 0.05194805194805195
               elif obj[2]<=2.723866729253481:
                  return 0.1865671641791045
               else:
                  return 0.07033639143730887
            elif obj[1]>44.797436015325665:
               # {"feature": "feature_2", "instances": 585, "metric_value": 0.0568, "depth": 5}
               if obj[2]<=3.6195608547008553:
                  return 0
               elif obj[2]>3.6195608547008553:
                  return 0.05504587155963303
               else:
                  return 0.020512820512820513
            else:
               return 0.05172413793103448
         elif obj[9]>308.6146710901211:
            return 0
         else:
            return 0.043016463090812536
      else:
         return 0.39436407474382157
   else:
      return 0.6498802360226675
