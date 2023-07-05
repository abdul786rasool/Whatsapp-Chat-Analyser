import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns




def main():
    st.sidebar.title("Whatsapp Chat Analyzer")
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        #fetch unique users
        user_list = df['user'].unique().tolist()
        user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0,"Overall")
        selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)
        if st.sidebar.button("Show Analysis"):
            num_messages,words, num_media_messages ,num_links = helper.fetch_stats(selected_user,df)
            st.title("Top Statistics")
            col1, col2 , col3, col4 = st.columns(4)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Media shared")
                st.title(num_media_messages)
            with col4:
                st.header("Links shared")
                st.title(num_links)

            #monthly_timeline
            st.title("Monthly Timeline")
            timeline=helper.monthly_timeline(selected_user,df)
            fig = plt.figure()
            sns.set_style('darkgrid')
            sns.lineplot(x=timeline['time'],y=timeline['message'],color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig = plt.figure(figsize=(10, 3))
            sns.set_style('whitegrid')
            sns.lineplot(x=daily_timeline['only_date'],y=daily_timeline['message'], color='purple')
            plt.xticks(rotation='vertical')
            plt.xlabel("date")
            st.pyplot(fig)

            #activity map
            st.title('Activity Map')
            col1,col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day=helper.week_activity_map(selected_user,df)
                fig = plt.figure()
                sns.set_style('ticks')
                pal=sns.cubehelix_palette(start=2, rot=0, dark=0.5, light=0.9, reverse=True)
                sns.barplot(x=busy_day.index,y=busy_day.values,palette=pal)
                plt.ylabel("messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month")
                busy_month=helper.month_activity_map(selected_user,df)
                fig=plt.figure()
                sns.set_style('ticks')
                pal=sns.cubehelix_palette(start=0, rot=0, dark=0.2, light=0.9, reverse=True)
                sns.barplot(x=busy_month.index,y=busy_month.values,palette=pal)
                plt.ylabel("messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig = plt.figure(figsize=(13,4))
            cmap = sns.color_palette("viridis", as_cmap=True)
            sns.heatmap(user_heatmap,cmap=cmap,square=True)
            st.pyplot(fig)

            #finding active users
            if(selected_user=='Overall'):
                st.title('Most active users')
                x,new_df=helper.most_busy_users(df)
                fig=plt.figure()

                col1,col2 = st.columns(2)

                with col1:
                    pal = sns.color_palette("cubehelix")
                    sns.barplot(x=x.index,y=x.values,palette=pal)
                    plt.xticks(rotation='vertical')
                    plt.ylabel('messages')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)
            #word cloud
            st.title('WordCloud')
            df_wc = helper.create_wordcloud(selected_user,df)
            fig,ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            #most common words
            st.title('Most Common Words')
            most_common_df=helper.most_common_words(selected_user,df)
            colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                      '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                      '#008080', '#e6beff', '#9a6324', '#fffac8', '#aaffc3',
                      '#808000', '#ffd8b1', '#808080', 'lightgreen', 'lightblue']
            # explosion

            fig = plt.figure()
            # Pie Chart
            plt.pie(most_common_df[1], labels=most_common_df[0], colors=colors,
                    autopct='%0.1f%%', pctdistance=0.9, labeldistance=1, rotatelabels=270, startangle=180,
                    counterclock=False)
            # draw circle
            centre_circle = plt.Circle((0, 0), 0.50, fc='white')
            fig2 = plt.gcf()
            # Adding Circle in Pie chart
            fig2.gca().add_artist(centre_circle)
            st.pyplot(fig)

            #emoji analysys

            emoji_df = helper.emoji_helper(selected_user,df)
            if(emoji_df.shape[0]):
                st.title("Emoji Analysis")
                col1,col2 =st.columns(2)
                with col1:
                    st.dataframe(emoji_df)
                with col2:
                    fig,ax = plt.subplots()
                    plt.rcParams['font.family'] = 'Segoe UI Emoji'
                    ax.pie(emoji_df[1].head(min(5,emoji_df.shape[0])),labels=emoji_df[0].head(min(5,emoji_df.shape[0])),autopct="%0.2f")
                    st.pyplot(fig)

            #birth_dates
            if(selected_user=='Overall'):
                birth_data = helper.birth_dates(df)
                if(birth_data.shape[0]):
                    st.title("Birth dates of some users.")
                    st.dataframe(birth_data)

            #Sentiment-analysis
            if (selected_user != 'Overall'):
                st.title("Sentiment Analysis")
                sentiment_data,number=helper.sentiment_analysis(selected_user,df)
                fig = plt.figure()
                sns.set_style('ticks')
                pal = sns.cubehelix_palette(start=0.5, rot=0, dark=0.2, light=0.9, reverse=True)
                sns.barplot(x=sentiment_data.index,y=sentiment_data.values,palette=pal)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
                st.header("Based on random "+str(number)+" messages.")
                st.text("Note : Sentiment Analysis give good results if messages \nare in hinglish (hindi or english or both).")
main()
