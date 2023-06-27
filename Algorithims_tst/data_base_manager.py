"""
Author: Joel Weber
Date: 20/06/2023
Description: Takes the lines as a list and saves them to a csv file 
"""
import pandas as pd
import csv
import os
import numpy as np
from icecream import ic



class data_base_manager(object):
    def __init__(self, path = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/saved_tests/saved_lines", data_name = 'test_db.csv'):
        # self.lines = lines
        self.save_path = os.path.join(path,data_name)
        if not self._check_for_existing_db():
            ic ("Making new database")            
            self.make_db()

    def _check_for_existing_db(self):
        # check if the file exists
        return os.path.exists(self.save_path)
    
    def make_db(self):
        # make a csv file at the following path
        assert ".csv" in self.save_path, "The data_name must be a csv file"
        

        with open (self.save_path, 'w', newline='') as file:
            writer = csv.writer(file)

        # df = pd.DataFrame()

    def update_csv(self,data):
        
        
        try:
            df = pd.read_csv(self.save_path)
            inpt = input("The database already exists. Would you like to clear it? (y/n)")
            if inpt == "y":
                self.make_db()
                df = pd.DataFrame()

        except (pd.errors.EmptyDataError):
            ic("The file is empty")
            df = pd.DataFrame()

        data = np.array(data, dtype=np.float32)
        # new_data = pd.DataFrame(data, index=["sl","int"],columns=[0,1],  dtype=np.float32)

        key =  0 if len(df.columns.tolist()) == 0 else self._find_max_index(df) + 1 
        ic(key, df.shape)
        # ic(len(df),df, df.columns.tolist(), df.head())
        # df[key] = new_data 

        # new_data = {f"sl{key}":data[:,0], f"int{key}" : data[:,1]}
        # new_data = np.array([data[:,0], data[:,1]],dtype=np.float32)
        
        if df.shape[0] == data.shape[0]:
            df[f"sl{key}"] = data[:,0]
            df[f"int{key}"] = data[:,1]
        else:
            df = self.match_data_len(df, data)
        ic(df)
        # df.dtype = np.float32
        # df = pd.concat([df,new_data],axis=0)
        df.to_csv(self.save_path,  header=str(key),index=int)

    def match_data_len(self, df, new_data):
        # make sure that the data is the same length as the data base
        dshp = df.shape
        # longest = max(max(val := [len(df[f"int{i}"].values.tolist()) for i in np.arange(0,int(df.columns.tolist()[-1][-1])+1)]), new_data.shape[0])

        new_df = pd.DataFrame()


        diff = abs(new_data.shape[0] - dshp[0])
        mt_space = [np.nan]*diff
        cur_ind = self._find_max_index(df) +1
        if dshp[0] < new_data.shape[0]:
            new_df[f"sl{cur_ind}"] = new_data[:,0]
            new_df[f"int{cur_ind}"] = new_data[:,1]
            ic(dshp)
            for i in np.arange(0,cur_ind):
                ic(i)
                new_df[f"sl{i}"] = df[f"sl{i}"].values.tolist() + mt_space
                new_df[f"int{i}"] = df[f"int{i}"].values.tolist() + mt_space
            ic(new_df)
            return new_df

        df[f"sl{cur_ind}"] = new_data[:,0].tolist() + mt_space
        df[f"int{cur_ind}"] = new_data[:,1].tolist() + mt_space

        return df

       
    def _find_max_index(self,df): 
        Vmax = 0
        for col in df.columns.tolist():
            if "Unnamed" in col:
                continue
            if int(col[-1]) > Vmax:
                Vmax = int(col[-1])
        return Vmax




    # the data base is not hashed with a string for advanced searching so only index or slice retieval is possible
    def read_data_base(self,ind=None):
        df = pd.read_csv(self.save_path) 
        
        cur_ind = self._find_max_index(df) +1
        rows = np.array([list(zip(df[f"sl{i}"].values.tolist(),df[f"int{i}"].values.tolist())) for i in np.arange(0,cur_ind)])
        # remove the nan values **note this cannot be convereted to a numpy array because the rows are not the same length.

        rem_nan = lambda row: row[~np.isnan(row)].reshape(-1,2).tolist()
        clean_rows = [rem_nan(row) for row in rows]
        
        if ind == None:
            return clean_rows

        try:
            return clean_rows[ind] 
        except TypeError:
            try:
                iter(ind)
            except TypeError:
                raise TypeError("The index must be an integer, slice or iterable of integers.")
            
            ret = clean_rows[slice(*ind)] if len(ind) == 2 else [clean_rows[i] for i in ind]
            return ret            

        except IndexError:
            raise IndexError("The index is out of range for the data base. Please ensure that the index is within range and is only an integer or slice.")

    def clear_data_base(self):
        self.make_db()
            

def test():
    # lines = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    lines = [
        [111,2],
        [3,4],
        [2,3],
        [43,2],
        [0,0]
    ]
    ln_offset = [
        [1,2],
        [12,3]
    ]
    dbm = data_base_manager()
    # dbm.update_csv(ln_offset)
    # dbm.update_csv(lines)
    ic(dbm.read_data_base())
    # ic(dbm.read_data_base(4))
    # ic(dbm.read_data_base([1,3]))
    # ic(dbm.read_data_base(slice(1,3)))
    # ic(dbm.read_data_base((1,2,3)))
    # dbm.clear_data_base()

def read_winter_wheat():
    dbm = data_base_manager(data_name='winter_wheat.csv')
    ic(dbm.read_data_base())
    # ic(dbm.read_data_base(2))
    return dbm


if __name__ == "__main__":
    test()
    # read_winter_wheat()