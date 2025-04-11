from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


# Function to train the model
def train_RFmodel(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Train the logistic regression model
    model = RandomForestClassifier(n_estimators=2,
                                    max_depth=2,
                                    max_features=8).fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open('models/RFmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, X_test_scaled, y_test
