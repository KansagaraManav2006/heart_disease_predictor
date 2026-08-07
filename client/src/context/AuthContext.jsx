import React, { useState, useEffect } from 'react';
import { AuthContext } from './AuthContextObject';
import { fetchMe, login, register, logout, getCSRFToken } from '../services/api';

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const reloadUser = async () => {
    try {
      await getCSRFToken();
      const currentUser = await fetchMe();
      setUser(currentUser);
    } catch (err) {
      console.error('Failed to load user session:', err);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reloadUser();
  }, []);

  const signIn = async ({ email, password }) => {
    const res = await login({ email, password });
    setUser(res.user);
    return res;
  };

  const signUp = async ({ email, password, role, fullName }) => {
    const res = await register({ email, password, role, fullName });
    return res;
  };

  const signOut = async () => {
    await logout();
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        signIn,
        signUp,
        signOut,
        reloadUser,
        isAuthenticated: !!user,
        isClinician: user?.role === 'CLINICIAN',
        isAdmin: user?.role === 'ADMIN',
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
