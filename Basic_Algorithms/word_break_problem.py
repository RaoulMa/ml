"""
Description: Word Break Problem
"""

# dictionary
dict = ["arrays", "dynamic", "heaps", "IDeserve", "learn", "learning", 
        "linked", "list", "platform", "programming", "stacks", "trees"]

def word_in_dict(string):
    for i in range(0, len(dict)):
        if string == dict[i]:
            return True
    return False

def has_valid_words(string):
    
    if len(string)==0:
        print('Exception: str is empty')
        return False
    if not isinstance(string, str):
        print('Exception: str not a string')
        return False
    
    valid_words = [False for i in range(len(string))]
    for i in range(0,len(string)):

        # first word in string
        if word_in_dict(string[0:i+1]):
            valid_words[i] = True;
        
        # i is the last character of the last identified word 
        # the next word begins at index i+1
        if valid_words[i]==True:
            
            # already reached end of string
            if i == (len(string)-1):
                print('valid_words' , valid_words)        
                return True
            
            for j in range(i+1, len(string)):
                if word_in_dict(string[i+1:j+1]):
                    valid_words[j] = True;
                    if j == (len(string)-1):
                        #print('valid_words' , valid_words)        
                        return True
 
    #print('valid_words' , valid_words)         
    return False

if __name__ == "__main__":
    print("dictionary: ", dict)
    string = "IDeservelearningplatform"
    
    print('string: ', string, type(string))
    if has_valid_words(string):
        print('yes, string contains only words from dictionary')
    else:
        print('no, string does not contain only words from dictionary')
        


