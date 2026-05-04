//Simulating multiple cycles, start 100, end 100, current when charging 27
T1 = BatteryCCCVSimlog.Battery_Table_Based.H.T.series.values;
Q1 = BatteryCCCVSimlog.Controlled_Temperature_Source.Q.series.values;
AT1 = BatteryCCCVSimlog.Controlled_Temperature_Source.A.T.series.values;
I1 = BatteryCCCVSimlog.Controlled_Current_Source.i.series.values;
V1 = BatteryCCCVSimlog.Controlled_Current_Source.v.series.values;
df1 = horzcat(T1, Q1, AT1, I1, V1);
df1 = array2table(df1);
colnames = ["T1", "Q1", "AT1", "I1", "V1"];
df1.Properties.VariableNames = colnames;
writetable(df1, "batteryCCCV_ordinary1_27.csv", "Delimiter", ",");

//Simulating multiple cycles, start 100, end 100, current when charging 54
T1 = BatteryCCCVSimlog.Battery_Table_Based.H.T.series.values;
Q1 = BatteryCCCVSimlog.Controlled_Temperature_Source.Q.series.values;
AT1 = BatteryCCCVSimlog.Controlled_Temperature_Source.A.T.series.values;
I1 = BatteryCCCVSimlog.Controlled_Current_Source.i.series.values;
V1 = BatteryCCCVSimlog.Controlled_Current_Source.v.series.values;
df1 = horzcat(T1, Q1, AT1, I1, V1);
df1 = array2table(df1);
colnames = ["T1", "Q1", "AT1", "I1", "V1"];
df1.Properties.VariableNames = colnames;
writetable(df1, "batteryCCCV_ordinary1_54.csv", "Delimiter", ",");


//Simulating multiple cycles, start 100, end 100, current when charging 81
T1 = BatteryCCCVSimlog.Battery_Table_Based.H.T.series.values;
Q1 = BatteryCCCVSimlog.Controlled_Temperature_Source.Q.series.values;
AT1 = BatteryCCCVSimlog.Controlled_Temperature_Source.A.T.series.values;
I1 = BatteryCCCVSimlog.Controlled_Current_Source.i.series.values;
V1 = BatteryCCCVSimlog.Controlled_Current_Source.v.series.values;
df1 = horzcat(T1, Q1, AT1, I1, V1);
df1 = array2table(df1);
colnames = ["T1", "Q1", "AT1", "I1", "V1"];
df1.Properties.VariableNames = colnames;
writetable(df1, "batteryCCCV_ordinary1_81.csv", "Delimiter", ",");


//Simulating multiple cycles, start 100, end 100, current when charging 108
T1 = BatteryCCCVSimlog.Battery_Table_Based.H.T.series.values;
Q1 = BatteryCCCVSimlog.Controlled_Temperature_Source.Q.series.values;
AT1 = BatteryCCCVSimlog.Controlled_Temperature_Source.A.T.series.values;
I1 = BatteryCCCVSimlog.Controlled_Current_Source.i.series.values;
V1 = BatteryCCCVSimlog.Controlled_Current_Source.v.series.values;
df1 = horzcat(T1, Q1, AT1, I1, V1);
df1 = array2table(df1);
colnames = ["T1", "Q1", "AT1", "I1", "V1"];
df1.Properties.VariableNames = colnames;
writetable(df1, "batteryCCCV_ordinary1_108.csv", "Delimiter", ",");


//Simulating multiple cycles, start 100, end 100, current when charging 135
T1 = BatteryCCCVSimlog.Battery_Table_Based.H.T.series.values;
Q1 = BatteryCCCVSimlog.Controlled_Temperature_Source.Q.series.values;
AT1 = BatteryCCCVSimlog.Controlled_Temperature_Source.A.T.series.values;
I1 = BatteryCCCVSimlog.Controlled_Current_Source.i.series.values;
V1 = BatteryCCCVSimlog.Controlled_Current_Source.v.series.values;
df1 = horzcat(T1, Q1, AT1, I1, V1);
df1 = array2table(df1);
colnames = ["T1", "Q1", "AT1", "I1", "V1"];
df1.Properties.VariableNames = colnames;
writetable(df1, "batteryCCCV_ordinary1_135.csv", "Delimiter", ",");
